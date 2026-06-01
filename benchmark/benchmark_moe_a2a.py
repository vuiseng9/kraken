"""
Benchmark MoE dispatch/combine at the splits-aware comm-and-permute layer.

Backends:
    torchtitan_eager    Eager path used by TorchTitan's AllToAllTokenDispatcher
    pytorch_symm_mem    PyTorch SymmetricMemory ops with NVSHMEM backend
    kraken_triton       Triton symm-mem kernels in kraken.comm

Excluded from every backend because this benchmark starts at the comm/layout boundary:
    - Router/local packing before dispatch: histc / argsort / x[indices]
    - Routing-score multiply, wherever the model config applies it
    - Expert compute between dispatch and combine
    - Final combine epilogue: shared_experts / scatter_add

Some asymmetries between impl worth nothing:
    - Dispatch: TorchTitan does a fresh count exchange plus D2H split-list
      materialization; the symm-mem backends keep this on device.
    - Combine: TorchTitan reuses CPU split metadata saved from dispatch; the
      symm-mem backends do a small split/offset exchange each call.
"""

# ruff: noqa: I001
import argparse
from collections import defaultdict
from collections.abc import Callable
import csv
from dataclasses import asdict, dataclass
import os
import sys

from tabulate import tabulate
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.distributed._functional_collectives import (
    all_to_all_single as fc_all_to_all_single,
    all_to_all_single_autograd as fc_all_to_all_single_autograd,
)

# Add the kraken directory to the Python path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import kraken
from kraken._logging import benchmark_with_event


Fn = Callable[[], object]
BACKENDS = ["torchtitan_eager", "pytorch_symm_mem", "kraken_triton"]
SYMM_MEM_REFS: list[torch.Tensor] = []


def symm_empty(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    *,
    min_numel: int = 32,
) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    if numel >= min_numel:
        tensor = symm_mem.empty(shape, dtype=dtype, device=device)
        SYMM_MEM_REFS.append(tensor)
        return tensor

    flat = symm_mem.empty((min_numel,), dtype=dtype, device=device)
    SYMM_MEM_REFS.append(flat)
    return flat[:numel].view(shape)


def clone_symm_mem_tensor(tensor: torch.Tensor) -> torch.Tensor:
    out = symm_empty(tuple(tensor.shape), dtype=tensor.dtype, device=tensor.device)
    out.copy_(tensor)
    return out


def exact_random_splits(
    total: int,
    nsplits: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    assert total >= 0
    assert nsplits > 0
    weights = torch.rand(nsplits, device=device, generator=generator)
    weights = weights / weights.sum()
    raw = weights * total
    splits = torch.floor(raw).to(torch.int64)
    remainder = total - int(splits.sum().item())
    if remainder > 0:
        order = torch.argsort(raw - splits.to(raw.dtype), descending=True)
        splits[order[:remainder]] += 1
    return splits


def make_splits_offsets(splits: torch.Tensor) -> torch.Tensor:
    offsets = splits.cumsum(0) - splits
    splits_offsets = symm_empty((2, splits.numel()), torch.int64, splits.device)
    splits_offsets[0].copy_(splits)
    splits_offsets[1].copy_(offsets)
    return splits_offsets


def all_to_all_splits(splits: torch.Tensor) -> torch.Tensor:
    out_splits = torch.empty_like(splits)
    dist.all_to_all_single(out_splits, splits)
    return out_splits


def output_offsets_from_dispatch_splits(
    output_splits_rank_major: torch.Tensor,
    world_size: int,
    ne: int,
    align: int,
) -> torch.Tensor:
    split_rows = output_splits_rank_major.reshape(world_size, ne).t().contiguous()
    split_list = split_rows.tolist()
    for expert in range(ne):
        expert_sum = sum(split_list[expert])
        aligned_sum = (expert_sum + align - 1) // align * align
        aligned_sum = max(aligned_sum, align)
        split_list[expert][-1] += aligned_sum - expert_sum

    padded_splits = torch.tensor(
        split_list,
        device=output_splits_rank_major.device,
    ).reshape(-1)
    return torch.cumsum(padded_splits, dim=0) - padded_splits


def permute_indices_from_rank_major_splits(
    rank_major_splits: torch.Tensor,
    world_size: int,
    ne: int,
) -> torch.Tensor:
    t_mat = rank_major_splits.view(world_size, ne)
    input_starts = (rank_major_splits.cumsum(0) - rank_major_splits).view(
        world_size,
        ne,
    )

    segment_lens = t_mat.t().reshape(-1)
    input_starts = input_starts.t().reshape(-1)
    total = segment_lens.sum()

    seg_ids = torch.arange(segment_lens.shape[0], device=rank_major_splits.device)
    seg_ids = seg_ids.repeat_interleave(segment_lens)
    output_starts = segment_lens.cumsum(0) - segment_lens
    return input_starts[seg_ids] + torch.arange(
        total,
        device=rank_major_splits.device,
    ) - output_starts[seg_ids]


def compact_indices_from_splits_offsets(
    splits: torch.Tensor,
    offsets: torch.Tensor,
) -> torch.Tensor:
    ranges = []
    for split_id in range(splits.numel()):
        split = int(splits[split_id].item())
        offset = int(offsets[split_id].item())
        if split > 0:
            ranges.append(
                torch.arange(
                    offset,
                    offset + split,
                    dtype=torch.int64,
                    device=splits.device,
                )
            )
    if ranges:
        return torch.cat(ranges)
    return torch.empty(0, dtype=torch.int64, device=splits.device)


def assert_segmented_close(
    actual: torch.Tensor,
    actual_splits_offsets: torch.Tensor,
    expected: torch.Tensor,
    expected_splits_offsets: torch.Tensor,
) -> None:
    torch.testing.assert_close(actual_splits_offsets, expected_splits_offsets)
    for split_id in range(actual_splits_offsets.shape[1]):
        split = int(actual_splits_offsets[0, split_id].item())
        actual_offset = int(actual_splits_offsets[1, split_id].item())
        expected_offset = int(expected_splits_offsets[1, split_id].item())
        torch.testing.assert_close(
            actual[actual_offset : actual_offset + split],
            expected[expected_offset : expected_offset + split],
        )


def assert_compact_matches_segmented(
    compact: torch.Tensor,
    compact_splits: torch.Tensor,
    segmented: torch.Tensor,
    segmented_splits_offsets: torch.Tensor,
) -> None:
    compact_offsets = compact_splits.cumsum(0) - compact_splits
    for split_id in range(compact_splits.numel()):
        split = int(compact_splits[split_id].item())
        compact_offset = int(compact_offsets[split_id].item())
        segmented_offset = int(segmented_splits_offsets[1, split_id].item())
        torch.testing.assert_close(
            compact[compact_offset : compact_offset + split],
            segmented[segmented_offset : segmented_offset + split],
        )


def max_reduce_us(latency_us: float, device: torch.device) -> float:
    latency = torch.tensor([latency_us], dtype=torch.float64, device=device)
    dist.all_reduce(latency, op=dist.ReduceOp.MAX)
    return float(latency.item())


def bandwidth_gbs(bytes_per_rank: int, latency_us: float) -> float:
    return bytes_per_rank / (latency_us * 1e-6) / 1e9


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    tokens_per_rank: int
    ne: int
    hidden: int
    align: int
    top_k: int
    dtype: torch.dtype
    backends: list[str]
    baseline_backend: str
    device: torch.device
    seed: int
    num_warps: int
    copy_block_size: int
    max_num_blocks: int
    check_correctness: bool
    warmup_iters: int
    benchmark_iters: int

    @property
    def bytes_per_rank(self) -> int:
        return (
            self.tokens_per_rank
            * self.top_k
            * self.hidden
            * torch.empty((), dtype=self.dtype).element_size()
        )

    def asdict(self) -> dict[str, object]:
        d = asdict(self)
        d.pop("backends", None)
        d.pop("baseline_backend", None)
        d.pop("device", None)
        d.pop("seed", None)
        d.pop("check_correctness", None)
        d["dtype"] = str(self.dtype).replace("torch.", "")
        d["bytes_per_rank"] = self.bytes_per_rank
        d["world_size"] = dist.get_world_size()
        return d


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    results: dict[str, float]

    def asdict(self) -> dict[str, object]:
        return {**self.config.asdict(), **self.results}


def generate_experiment_configs(
    args: argparse.Namespace,
    device: torch.device,
) -> list[ExperimentConfig]:
    modes = ["dispatch", "combine"] if args.mode == "all" else [args.mode]
    configs = []
    for mode in modes:
        for tokens_per_rank in args.tokens_per_rank:
            for ne in args.ne:
                for hidden in args.hidden:
                    configs.append(
                        ExperimentConfig(
                            mode=mode,
                            tokens_per_rank=tokens_per_rank,
                            ne=ne,
                            hidden=hidden,
                            align=args.align,
                            top_k=args.top_k,
                            dtype=args.dtype,
                            backends=args.backend,
                            baseline_backend=args.backend[0],
                            device=device,
                            seed=args.seed,
                            num_warps=args.num_warps,
                            copy_block_size=args.copy_block_size,
                            max_num_blocks=args.max_num_blocks,
                            check_correctness=args.check_correctness,
                            warmup_iters=args.warmup_iters,
                            benchmark_iters=args.benchmark_iters,
                        )
                    )
    return configs


def make_base_values(config: ExperimentConfig) -> torch.Tensor:
    rank = dist.get_rank()
    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed + 1009 * rank)
    return torch.randn(
        (config.tokens_per_rank * config.top_k, config.hidden),
        dtype=config.dtype,
        device=config.device,
        generator=generator,
    )


def make_base_splits(config: ExperimentConfig, salt: int) -> torch.Tensor:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    generator = torch.Generator(device=config.device)
    generator.manual_seed(config.seed + salt + 101 * rank)
    return exact_random_splits(
        config.tokens_per_rank * config.top_k,
        config.ne * world_size,
        config.device,
        generator,
    )


def dispatch_benchmarks(config: ExperimentConfig) -> dict[str, Fn]:
    world_size = dist.get_world_size()
    nsplits = config.ne * world_size
    max_out_tokens = config.tokens_per_rank * config.top_k * world_size + config.ne * config.align

    base_values = make_base_values(config)
    base_splits = make_base_splits(config, salt=0)

    triton_inp = clone_symm_mem_tensor(base_values)
    triton_splits = clone_symm_mem_tensor(base_splits)
    symm_inp = clone_symm_mem_tensor(base_values)
    symm_splits = clone_symm_mem_tensor(base_splits)
    eager_inp = base_values

    triton_out = symm_empty(
        (max_out_tokens, config.hidden),
        dtype=config.dtype,
        device=config.device,
    )
    triton_splits_offsets = symm_empty(
        (2, nsplits),
        dtype=torch.int64,
        device=config.device,
    )
    symm_out = symm_empty(
        (max_out_tokens, config.hidden),
        dtype=config.dtype,
        device=config.device,
    )
    symm_splits_offsets = symm_empty(
        (2, nsplits),
        dtype=torch.int64,
        device=config.device,
    )

    output_splits_rank_major = all_to_all_splits(base_splits)
    output_tokens = int(output_splits_rank_major.sum().item())
    output_splits_expert_major = (
        output_splits_rank_major.reshape(world_size, config.ne).t().reshape(-1)
    )
    output_offsets_expert_major = output_offsets_from_dispatch_splits(
        output_splits_rank_major,
        world_size,
        config.ne,
        config.align,
    )
    eager_expert_major = torch.empty(
        (output_tokens, config.hidden),
        dtype=config.dtype,
        device=config.device,
    )

    def kraken_triton() -> torch.Tensor:
        kraken.comm.all_to_all_vdev_2d(
            triton_inp,
            triton_out,
            triton_splits,
            triton_splits_offsets,
            major_align=config.align,
            num_warps=config.num_warps,
            copy_block_size=config.copy_block_size,
            max_num_blocks=config.max_num_blocks,
        )
        return triton_out

    def pytorch_symm_mem() -> torch.Tensor:
        torch.ops.symm_mem.all_to_all_vdev_2d(
            symm_inp,
            symm_out,
            symm_splits,
            symm_splits_offsets,
            dist.group.WORLD.group_name,
            major_align=config.align,
        )
        return symm_out

    def torchtitan_eager() -> torch.Tensor:
        num_tokens_per_expert_group = fc_all_to_all_single(
            base_splits,
            None,
            None,
            group=dist.group.WORLD,
        )
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
            num_tokens_per_expert_group
        )
        input_splits = (
            base_splits.view(world_size, config.ne)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        output_splits = (
            num_tokens_per_expert_group.view(world_size, config.ne)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        input_splits_list = input_splits.tolist()
        output_splits_list = output_splits.tolist()

        routed_input = fc_all_to_all_single_autograd(
            eager_inp,
            output_splits_list,
            input_splits_list,
            dist.group.WORLD,
        )
        indices = permute_indices_from_rank_major_splits(
            num_tokens_per_expert_group,
            world_size,
            config.ne,
        )
        torch.index_select(routed_input, 0, indices, out=eager_expert_major)
        return eager_expert_major

    benchmarks: dict[str, Fn] = {
        "torchtitan_eager": torchtitan_eager,
        "pytorch_symm_mem": pytorch_symm_mem,
        "kraken_triton": kraken_triton,
    }

    if config.check_correctness:
        kraken_triton()
        pytorch_symm_mem()
        assert_segmented_close(
            triton_out,
            triton_splits_offsets,
            symm_out,
            symm_splits_offsets,
        )
        eager_out = torchtitan_eager()
        expected_splits_offsets = torch.stack(
            (output_splits_expert_major, output_offsets_expert_major)
        )
        assert_compact_matches_segmented(
            eager_out,
            output_splits_expert_major,
            symm_out,
            expected_splits_offsets,
        )

    return benchmarks


def combine_benchmarks(config: ExperimentConfig) -> dict[str, Fn]:
    world_size = dist.get_world_size()
    nsplits = config.ne * world_size
    max_out_tokens = config.tokens_per_rank * config.top_k * world_size

    base_values = make_base_values(config)
    base_splits_expert_major = make_base_splits(config, salt=12345)
    base_offsets_expert_major = (
        base_splits_expert_major.cumsum(0) - base_splits_expert_major
    )

    triton_inp = clone_symm_mem_tensor(base_values)
    triton_splits_offsets_in = make_splits_offsets(base_splits_expert_major)
    symm_inp = clone_symm_mem_tensor(base_values)
    symm_splits_offsets_in = make_splits_offsets(base_splits_expert_major)

    triton_out = symm_empty(
        (max_out_tokens, config.hidden),
        dtype=config.dtype,
        device=config.device,
    )
    triton_splits_offsets = symm_empty(
        (2, nsplits),
        dtype=torch.int64,
        device=config.device,
    )
    symm_out = symm_empty(
        (max_out_tokens, config.hidden),
        dtype=config.dtype,
        device=config.device,
    )
    symm_splits_offsets = symm_empty(
        (2, nsplits),
        dtype=torch.int64,
        device=config.device,
    )

    rank_major_splits = (
        base_splits_expert_major.reshape(config.ne, world_size).t().reshape(-1)
    )
    output_splits_rank_major = all_to_all_splits(rank_major_splits)
    send_split_sizes = (
        base_splits_expert_major.reshape(config.ne, world_size).sum(0).tolist()
    )
    recv_split_sizes = output_splits_rank_major.view(world_size, config.ne).sum(
        1
    ).tolist()

    compact_expert_major_indices = compact_indices_from_splits_offsets(
        base_splits_expert_major,
        base_offsets_expert_major,
    )
    compact_expert_major = torch.empty(
        (config.tokens_per_rank * config.top_k, config.hidden),
        dtype=config.dtype,
        device=config.device,
    )
    torch.index_select(
        base_values,
        0,
        compact_expert_major_indices,
        out=compact_expert_major,
    )

    unpermute_indices = permute_indices_from_rank_major_splits(
        rank_major_splits,
        world_size,
        config.ne,
    )
    eager_rank_major = torch.empty_like(compact_expert_major)

    def kraken_triton() -> torch.Tensor:
        kraken.comm.all_to_all_vdev_2d_offset(
            triton_inp,
            triton_out,
            triton_splits_offsets_in,
            triton_splits_offsets,
            num_warps=config.num_warps,
            copy_block_size=config.copy_block_size,
            max_num_blocks=config.max_num_blocks,
        )
        return triton_out

    def pytorch_symm_mem() -> torch.Tensor:
        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            symm_inp,
            symm_out,
            symm_splits_offsets_in,
            symm_splits_offsets,
            dist.group.WORLD.group_name,
        )
        return symm_out

    def torchtitan_eager() -> torch.Tensor:
        eager_rank_major[unpermute_indices, :] = compact_expert_major
        routed_output = fc_all_to_all_single_autograd(
            eager_rank_major,
            recv_split_sizes,
            send_split_sizes,
            dist.group.WORLD,
        )
        return torch.ops._c10d_functional.wait_tensor(routed_output)

    benchmarks: dict[str, Fn] = {
        "torchtitan_eager": torchtitan_eager,
        "pytorch_symm_mem": pytorch_symm_mem,
        "kraken_triton": kraken_triton,
    }

    if config.check_correctness:
        kraken_triton()
        pytorch_symm_mem()
        assert_segmented_close(
            triton_out,
            triton_splits_offsets,
            symm_out,
            symm_splits_offsets,
        )
        eager_out = torchtitan_eager()
        assert_compact_matches_segmented(
            eager_out,
            output_splits_rank_major,
            symm_out,
            symm_splits_offsets,
        )

    return benchmarks


def create_benchmarks(config: ExperimentConfig) -> dict[str, Fn]:
    if config.mode == "dispatch":
        return dispatch_benchmarks(config)
    if config.mode == "combine":
        return combine_benchmarks(config)
    raise NotImplementedError(config.mode)


def run_experiment(config: ExperimentConfig) -> dict[str, float]:
    all_benchmarks = create_benchmarks(config)

    results = {}
    for backend in config.backends:
        target_fn = all_benchmarks[backend]
        runtime_us = benchmark_with_event(
            target_fn,
            warmup_iters=config.warmup_iters,
            benchmark_iters=config.benchmark_iters,
            profile_ranks=[-1],
            flush_l2=True,
        )
        results[backend] = max_reduce_us(runtime_us, config.device)

    return results


def print_results(results: list[Experiment], save_path: str | None = None) -> None:
    table_data = defaultdict(list)

    for experiment in results:
        baseline_time = experiment.results[experiment.config.baseline_backend]
        min_time = float("inf")
        best_backend = experiment.config.baseline_backend

        for key, value in experiment.asdict().items():
            if key in experiment.config.backends:
                if value < min_time:
                    min_time = value
                    best_backend = key
                table_data[f"{key}_us"].append(value)
                table_data[f"{key}_GBs"].append(
                    bandwidth_gbs(experiment.config.bytes_per_rank, value)
                )
            else:
                table_data[key].append(value)

        table_data[f"Speedup over {experiment.config.baseline_backend}"].append(
            baseline_time / min_time
        )
        table_data["Best Backend"].append(best_backend)

    if dist.get_rank() == 0:
        print(tabulate(table_data, headers="keys", tablefmt="github", floatfmt=".3f"))

        if save_path is not None:
            with open(save_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=table_data.keys())
                writer.writeheader()
                for i in range(len(next(iter(table_data.values())))):
                    row = {k: v[i] for k, v in table_data.items()}
                    writer.writerow(row)
            print(f"\nResults saved to {save_path}")


def main(args: argparse.Namespace) -> None:
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    symm_mem.set_backend("NVSHMEM")
    dist.init_process_group("nccl", device_id=device)
    world_size = dist.get_world_size()
    for ne in args.ne:
        if args.top_k >= ne * world_size:
            if dist.get_rank() == 0:
                print(
                    f"Error: --top-k ({args.top_k}) must be smaller than "
                    f"ne * world_size ({ne} * {world_size} = {ne * world_size})"
                )
            dist.destroy_process_group()
            sys.exit(1)
    _torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    if _torch_ver <= (2, 11):
        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
    torch.manual_seed(args.seed + local_rank)

    configs = generate_experiment_configs(args, device)
    results = []
    for config in configs:
        results.append(Experiment(config, run_experiment(config)))

    print_results(results, args.save_path)
    dist.destroy_process_group()


if __name__ == "__main__":
    DBG_ATTACH = False
    if int(os.environ.get("DBG_ATTACH", "0")) == 1:
        DBG_ATTACH = True
        
    if DBG_ATTACH and int(os.environ.get("RANK", "0")) == 0:
        import debugpy
        debugpy.listen(("127.0.0.1", 9999))
        # optional (only when you want to pause immediately):
        print('\n\n\n\n\n#### Waiting for debugger attach...', flush=True)
        debugpy.wait_for_client()
    help_str = """
Run with torchrun
torchrun \\
--nnodes 1 --nproc-per-node 2 \\
--rdzv-backend c10d --rdzv-endpoint localhost:0 \\
--no_python python3 \\
benchmark/benchmark_moe_a2a.py
"""
    parser = argparse.ArgumentParser(
        description="Benchmark MoE dispatch/combine comm-layout paths. " + help_str
    )
    parser.add_argument("--mode", choices=["dispatch", "combine", "all"], default="all")
    parser.add_argument(
        "--backend",
        type=str,
        nargs="+",
        choices=BACKENDS,
        default=BACKENDS,
        help="Backends to benchmark. The first backend is used as baseline.",
    )
    parser.add_argument(
        "--tokens-per-rank",
        type=int,
        nargs="+",
        default=[4096],
    )
    parser.add_argument("--ne", type=int, nargs="+", default=[8])
    parser.add_argument("--hidden", type=int, nargs="+", default=[4096])
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--align", type=int, default=8)
    parser.add_argument("-dtype", "--dtype", type=str, default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--copy-block-size", type=int, default=2048)
    parser.add_argument("--max-num-blocks", type=int, default=1024)
    parser.add_argument("--warmup-iters", type=int, default=200)
    parser.add_argument("--benchmark-iters", type=int, default=25)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument(
        "--skip-correctness",
        dest="check_correctness",
        action="store_false",
    )
    parser.set_defaults(check_correctness=True)
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    if "LOCAL_RANK" not in os.environ:
        print("Error: LOCAL_RANK is not defined. Are you running with torchrun?")
        print(help_str)
        sys.exit(1)

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except ValueError:
        print("Error: LOCAL_RANK must be an integer. Are you running with torchrun?")
        print(help_str)
        sys.exit(1)

    main(args)
