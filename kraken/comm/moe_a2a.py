import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton
import triton.language as tl

from .. import _ptx_utils as ptx_utils

A2AV_TILE_SIZE = 32
NUM_TILES = 32


@triton.jit
def _moe_a2a_exchange_splits_offsets_kernel(
    in_splits_offsets,
    out_splits_offsets_tuple,
    signal_pad_ptrs,
    input_dim0: tl.constexpr,
    rank: tl.constexpr,
    world_size: tl.constexpr,
    ne: tl.constexpr,
    nsplits: tl.constexpr,
    rank_is_row_in: tl.constexpr,
    HAS_IN_OFFSETS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < nsplits

    input_splits = tl.load(in_splits_offsets + offsets, mask=mask, other=0)
    if HAS_IN_OFFSETS:
        input_offsets = tl.load(
            in_splits_offsets + nsplits + offsets, mask=mask, other=0
        )
    else:
        input_offsets = tl.cumsum(input_splits, 0) - input_splits
        total_input = tl.sum(input_splits, 0)
        tl.device_assert(total_input <= input_dim0, "sum of splits exceeds input dim")

    if rank_is_row_in:
        peers = offsets // ne
        experts = offsets - peers * ne
        dst_offsets = experts * world_size + rank
    else:
        peers = offsets % world_size
        experts = offsets // world_size
        dst_offsets = rank * ne + experts

    tl.device_assert(tl.min(input_splits, 0) >= 0, "split value is negative")

    for peer in tl.static_range(0, world_size):
        peer_mask = mask & (peers == peer)
        peer_out_splits_offsets = out_splits_offsets_tuple[peer]
        tl.store(
            peer_out_splits_offsets + dst_offsets,
            input_splits,
            mask=peer_mask,
        )
        tl.store(
            peer_out_splits_offsets + nsplits + dst_offsets,
            input_offsets,
            mask=peer_mask,
        )

    ptx_utils.symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
        hasSubsequentMemAccess=True,
    )


@triton.jit
def _major_length(output_splits, major, minor_size: tl.constexpr, major_align: tl.constexpr):
    length = tl.full((), 0, tl.int64)
    for minor in tl.static_range(0, minor_size):
        length += tl.load(output_splits + major * minor_size + minor)

    if major_align != 0:
        length = ((length + major_align - 1) // major_align) * major_align
        length = tl.maximum(length, major_align)

    return length


@triton.jit
def _output_offset(
    output_splits,
    row,
    col,
    minor_size: tl.constexpr,
    major_size: tl.constexpr,
    major_align: tl.constexpr,
):
    offset = tl.full((), 0, tl.int64)

    for major in tl.static_range(0, major_size):
        major_len = _major_length(output_splits, major, minor_size, major_align)
        offset += tl.where(major < row, major_len, 0)

    for minor in tl.static_range(0, minor_size):
        split = tl.load(output_splits + row * minor_size + minor)
        offset += tl.where(minor < col, split, 0)

    return offset


@triton.jit
def _moe_a2a_copy_kernel(
    input_tuple,
    out,
    out_splits_offsets,
    row_stride: tl.constexpr,
    nsplits: tl.constexpr,
    world_size: tl.constexpr,
    minor_size: tl.constexpr,
    major_size: tl.constexpr,
    major_align: tl.constexpr,
    rank_is_row_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_splits = out_splits_offsets
    source_offsets = out_splits_offsets + nsplits
    offsets = tl.arange(0, BLOCK_SIZE)

    eid = tl.program_id(axis=0)
    tile_id = tl.program_id(axis=1)
    tiles_per_split = tl.num_programs(axis=1)

    row = eid // minor_size
    col = eid - row * minor_size
    peer = row if rank_is_row_out else col

    split = tl.load(output_splits + eid)
    source_offset = tl.load(source_offsets + eid)
    write_offset = _output_offset(
        output_splits,
        row,
        col,
        minor_size,
        major_size,
        major_align,
    )
    copy_numel = split * row_stride

    block_start = tile_id * BLOCK_SIZE
    tile_stride = tiles_per_split * BLOCK_SIZE
    while block_start < copy_numel:
        copy_offsets = block_start + offsets
        copy_mask = copy_offsets < copy_numel
        for peer_idx in tl.static_range(0, world_size):
            peer_mask = copy_mask & (peer == peer_idx)
            peer_input = input_tuple[peer_idx]
            data = tl.load(
                peer_input + source_offset * row_stride + copy_offsets,
                mask=peer_mask,
            )
            tl.store(
                out + write_offset * row_stride + copy_offsets,
                data,
                mask=peer_mask,
            )
        block_start += tile_stride


@triton.jit
def _moe_a2a_finish_sync_kernel(
    signal_pad_ptrs,
    rank: tl.constexpr,
    world_size: tl.constexpr,
):
    ptx_utils.symm_mem_sync(
        signal_pad_ptrs,
        None,
        rank,
        world_size,
        hasPreviousMemAccess=True,
    )


@triton.jit
def _moe_a2a_finalize_offsets_kernel(
    out_splits_offsets,
    nsplits: tl.constexpr,
    minor_size: tl.constexpr,
    major_size: tl.constexpr,
    major_align: tl.constexpr,
):
    output_splits = out_splits_offsets
    output_offsets = out_splits_offsets + nsplits

    eid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)

    while eid < nsplits:
        row = eid // minor_size
        col = eid - row * minor_size
        write_offset = _output_offset(
            output_splits,
            row,
            col,
            minor_size,
            major_size,
            major_align,
        )
        tl.store(output_offsets + eid, write_offset)
        eid += grid_size


def _group_arg(group: dist.ProcessGroup | str | None) -> dist.ProcessGroup | str:
    return dist.group.WORLD if group is None else group


def _check_common_tensors(
    inp: torch.Tensor,
    out: torch.Tensor,
    out_splits_offsets: torch.Tensor,
) -> None:
    assert inp.is_cuda and out.is_cuda and out_splits_offsets.is_cuda, (
        "all tensors must be CUDA tensors"
    )
    assert inp.is_contiguous() and out.is_contiguous(), (
        "input and out must be contiguous"
    )
    assert out_splits_offsets.is_contiguous(), "out_splits_offsets must be contiguous"
    assert inp.dtype == out.dtype, "input and out must have the same dtype"
    assert inp.stride(0) == out.stride(0), (
        "input and out must have the same dim-0 stride"
    )
    assert out_splits_offsets.dtype == torch.int64, (
        "out_splits_offsets must be int64"
    )
    assert out_splits_offsets.dim() == 2 and out_splits_offsets.shape[0] == 2, (
        "out_splits_offsets must have shape (2, nsplits)"
    )


def _launch_moe_a2a(
    inp: torch.Tensor,
    out: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    *,
    group: dist.ProcessGroup | str | None,
    has_in_offsets: bool,
    rank_is_row_in: bool,
    rank_is_row_out: bool,
    major_align: int,
    num_warps: int,
    exchange_block_size: int,
    copy_block_size: int,
    max_num_blocks: int,
) -> None:
    _check_common_tensors(inp, out, out_splits_offsets)
    assert in_splits_offsets.is_cuda and in_splits_offsets.is_contiguous(), (
        "in_splits_offsets must be a contiguous CUDA tensor"
    )
    assert in_splits_offsets.dtype == torch.int64, "splits and offsets must be int64"
    assert major_align >= 0, "major_align must be non-negative"
    assert max_num_blocks > 0, "max_num_blocks must be positive"

    group_arg = _group_arg(group)

    input_hdl = symm_mem.rendezvous(inp, group=group_arg)
    symm_mem.rendezvous(out, group=group_arg)
    symm_mem.rendezvous(in_splits_offsets, group=group_arg)
    out_splits_offsets_hdl = symm_mem.rendezvous(out_splits_offsets, group=group_arg)

    rank = input_hdl.rank
    world_size = input_hdl.world_size
    assert world_size <= A2AV_TILE_SIZE, (
        f"world_size must be <= {A2AV_TILE_SIZE}, got {world_size}"
    )

    if has_in_offsets:
        assert in_splits_offsets.dim() == 2 and in_splits_offsets.shape[0] == 2, (
            "in_splits_offsets must have shape (2, nsplits)"
        )
        nsplits = in_splits_offsets.shape[1]
    else:
        assert in_splits_offsets.dim() == 1, "in_splits must be 1D"
        nsplits = in_splits_offsets.numel()

    assert out_splits_offsets.shape == (2, nsplits), (
        "out_splits_offsets shape must match the split count"
    )
    assert nsplits % world_size == 0, "nsplits must be a multiple of world_size"
    ne = nsplits // world_size

    if rank_is_row_in:
        assert ne <= NUM_TILES, f"number of experts per rank must be <= {NUM_TILES}"
    else:
        assert ne <= A2AV_TILE_SIZE, (
            f"number of experts per rank must be <= {A2AV_TILE_SIZE}"
        )

    if rank_is_row_out:
        minor_size = ne
        major_size = world_size
    else:
        minor_size = world_size
        major_size = ne

    if rank_is_row_out:
        assert world_size <= NUM_TILES, f"world_size must be <= {NUM_TILES}"
    else:
        assert ne <= NUM_TILES, f"number of experts per rank must be <= {NUM_TILES}"

    assert inp.dim() >= 1, "input must have at least one dimension"
    assert out.dim() == inp.dim(), "out must have the same rank as input"
    assert tuple(inp.shape[1:]) == tuple(out.shape[1:]), (
        "input and out trailing dimensions must match"
    )

    input_tuple = tuple(
        input_hdl.get_buffer(i, tuple(inp.shape), inp.dtype)
        for i in range(world_size)
    )
    out_splits_offsets_tuple = tuple(
        out_splits_offsets_hdl.get_buffer(
            i,
            tuple(out_splits_offsets.shape),
            out_splits_offsets.dtype,
        )
        for i in range(world_size)
    )

    _moe_a2a_exchange_splits_offsets_kernel[(1,)](
        in_splits_offsets,
        out_splits_offsets_tuple,
        out_splits_offsets_hdl.signal_pad_ptrs_dev,
        input_dim0=inp.shape[0],
        rank=rank,
        world_size=world_size,
        ne=ne,
        nsplits=nsplits,
        rank_is_row_in=rank_is_row_in,
        HAS_IN_OFFSETS=has_in_offsets,
        BLOCK_SIZE=exchange_block_size,
        num_warps=num_warps,
    )

    # Use a fixed CTA budget per split so large hidden dimensions do not serialize
    # through a single program.
    tiles_per_split = max(1, max_num_blocks // nsplits)
    offset_num_blocks = min(nsplits, max_num_blocks)
    _moe_a2a_copy_kernel[(nsplits, tiles_per_split)](
        input_tuple,
        out,
        out_splits_offsets,
        row_stride=inp.stride(0),
        nsplits=nsplits,
        world_size=world_size,
        minor_size=minor_size,
        major_size=major_size,
        major_align=major_align,
        rank_is_row_out=rank_is_row_out,
        BLOCK_SIZE=copy_block_size,
        num_warps=num_warps,
    )

    _moe_a2a_finalize_offsets_kernel[(offset_num_blocks,)](
        out_splits_offsets,
        nsplits=nsplits,
        minor_size=minor_size,
        major_size=major_size,
        major_align=major_align,
        num_warps=num_warps,
    )

    # One trailing sync is enough to protect the symmetric input buffer from
    # being overwritten while a peer is still reading it.
    _moe_a2a_finish_sync_kernel[(1,)](
        input_hdl.signal_pad_ptrs_dev,
        rank=rank,
        world_size=world_size,
        num_warps=num_warps,
    )


def _make_output_like_input(inp: torch.Tensor, max_out_tokens: int) -> torch.Tensor:
    out_shape = (max_out_tokens, *tuple(inp.shape[1:]))
    return symm_mem.empty(out_shape, dtype=inp.dtype, device=inp.device)


def moe_a2a_dispatch(
    inp: torch.Tensor,
    in_splits: torch.Tensor,
    *,
    max_out_tokens: int,
    major_align: int = 1,
    group: dist.ProcessGroup | str | None = None,
    num_warps: int = 8,
    exchange_block_size: int = 1024,
    copy_block_size: int = 2048,
    max_num_blocks: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Dispatch MoE tokens from rank-major input layout to expert-major output.

    This matches ``torch.ops.symm_mem.all_to_all_vdev_2d``. ``in_splits`` is a
    1D int64 tensor with ``ne * world_size`` elements in rank-major order. The
    returned ``out_splits_offsets`` has shape ``(2, ne * world_size)`` where row
    0 contains expert-major output splits and row 1 contains expert-major output
    offsets.
    """
    assert max_out_tokens >= 0, "max_out_tokens must be non-negative"
    assert major_align > 0, "major_align must be positive"

    out = _make_output_like_input(inp, max_out_tokens)
    out_splits_offsets = symm_mem.empty(
        (2, in_splits.numel()),
        dtype=torch.int64,
        device=in_splits.device,
    )

    all_to_all_vdev_2d(
        inp,
        out,
        in_splits,
        out_splits_offsets,
        group=group,
        major_align=major_align,
        num_warps=num_warps,
        exchange_block_size=exchange_block_size,
        copy_block_size=copy_block_size,
        max_num_blocks=max_num_blocks,
    )
    return out, out_splits_offsets


def all_to_all_vdev_2d(
    inp: torch.Tensor,
    out: torch.Tensor,
    in_splits: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    *,
    group: dist.ProcessGroup | str | None = None,
    major_align: int = 1,
    num_warps: int = 8,
    exchange_block_size: int = 1024,
    copy_block_size: int = 2048,
    max_num_blocks: int = 1024,
) -> None:
    assert major_align > 0, "major_align must be positive"
    _launch_moe_a2a(
        inp,
        out,
        in_splits,
        out_splits_offsets,
        group=group,
        has_in_offsets=False,
        rank_is_row_in=True,
        rank_is_row_out=False,
        major_align=major_align,
        num_warps=num_warps,
        exchange_block_size=exchange_block_size,
        copy_block_size=copy_block_size,
        max_num_blocks=max_num_blocks,
    )


def moe_a2a_combine(
    inp: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    *,
    max_out_tokens: int,
    group: dist.ProcessGroup | str | None = None,
    num_warps: int = 8,
    exchange_block_size: int = 1024,
    copy_block_size: int = 2048,
    max_num_blocks: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Combine MoE tokens from expert-major input layout back to rank-major output.

    This matches ``torch.ops.symm_mem.all_to_all_vdev_2d_offset``. The
    ``in_splits_offsets`` tensor has shape ``(2, ne * world_size)`` where row 0
    contains input splits and row 1 contains input offsets. The returned
    ``out_splits_offsets`` has the same shape, with rank-major output splits and
    output offsets.
    """
    assert max_out_tokens >= 0, "max_out_tokens must be non-negative"
    assert in_splits_offsets.dim() == 2 and in_splits_offsets.shape[0] == 2, (
        "in_splits_offsets must have shape (2, nsplits)"
    )

    out = _make_output_like_input(inp, max_out_tokens)
    out_splits_offsets = symm_mem.empty(
        tuple(in_splits_offsets.shape),
        dtype=torch.int64,
        device=in_splits_offsets.device,
    )

    all_to_all_vdev_2d_offset(
        inp,
        out,
        in_splits_offsets,
        out_splits_offsets,
        group=group,
        num_warps=num_warps,
        exchange_block_size=exchange_block_size,
        copy_block_size=copy_block_size,
        max_num_blocks=max_num_blocks,
    )
    return out, out_splits_offsets


def all_to_all_vdev_2d_offset(
    inp: torch.Tensor,
    out: torch.Tensor,
    in_splits_offsets: torch.Tensor,
    out_splits_offsets: torch.Tensor,
    *,
    group: dist.ProcessGroup | str | None = None,
    num_warps: int = 8,
    exchange_block_size: int = 1024,
    copy_block_size: int = 2048,
    max_num_blocks: int = 1024,
) -> None:
    """
    In-place combine wrapper matching
    ``torch.ops.symm_mem.all_to_all_vdev_2d_offset``.
    """
    _launch_moe_a2a(
        inp,
        out,
        in_splits_offsets,
        out_splits_offsets,
        group=group,
        has_in_offsets=True,
        rank_is_row_in=False,
        rank_is_row_out=True,
        major_align=0,
        num_warps=num_warps,
        exchange_block_size=exchange_block_size,
        copy_block_size=copy_block_size,
        max_num_blocks=max_num_blocks,
    )
