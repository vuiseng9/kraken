import os
import sys
import unittest

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import kraken


def _nvshmem_available() -> bool:
    return hasattr(symm_mem, "is_nvshmem_available") and symm_mem.is_nvshmem_available()


@instantiate_parametrized_tests
@unittest.skipIf(not _nvshmem_available(), "test_moe_a2a requires NVSHMEM")
class TritonMoeA2ATest(MultiProcessTestCase):
    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    @property
    def world_size(self) -> int:
        return 2

    @property
    def device(self) -> torch.device:
        return torch.device(f"cuda:{self.rank}")

    def _init_process(self) -> None:
        torch.cuda.set_device(self.device)
        symm_mem.set_backend("NVSHMEM")
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        torch.manual_seed(42 + self.rank)

    def _symm_copy(self, tensor: torch.Tensor) -> torch.Tensor:
        out = symm_mem.empty(
            tuple(tensor.shape),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        out.copy_(tensor)
        return out

    def _make_input(self, max_tokens: int, hidden: int, dtype: torch.dtype):
        values = torch.arange(
            max_tokens * hidden,
            dtype=torch.float32,
            device=self.device,
        ).reshape(max_tokens, hidden)
        values = values + self.rank * 100_000
        return self._symm_copy(values.to(dtype))

    def _assert_segmented_close(
        self,
        actual: torch.Tensor,
        actual_splits_offsets: torch.Tensor,
        expected: torch.Tensor,
        expected_splits_offsets: torch.Tensor,
    ) -> None:
        torch.testing.assert_close(actual_splits_offsets, expected_splits_offsets)
        nsplits = actual_splits_offsets.shape[1]
        for split_id in range(nsplits):
            split = actual_splits_offsets[0, split_id].item()
            actual_offset = actual_splits_offsets[1, split_id].item()
            expected_offset = expected_splits_offsets[1, split_id].item()
            torch.testing.assert_close(
                actual[actual_offset : actual_offset + split],
                expected[expected_offset : expected_offset + split],
            )

    @skip_if_lt_x_gpu(2)
    @parametrize("align", [1, 8, 16])
    def test_dispatch_matches_nvshmem_vdev_2d(self, align: int) -> None:
        self._init_process()

        ne = 4
        hidden = 16
        k = 6
        nsplits = ne * self.world_size
        splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=self.device)
        max_in_tokens = k * nsplits
        max_out_tokens = max_in_tokens * self.world_size + ne * align

        kraken_inp = self._make_input(max_in_tokens, hidden, torch.float32)
        kraken_splits = self._symm_copy(splits)
        ref_inp = self._make_input(max_in_tokens, hidden, torch.float32)
        ref_splits = self._symm_copy(splits)
        ref_out = symm_mem.empty(
            (max_out_tokens, hidden),
            dtype=torch.float32,
            device=self.device,
        )
        ref_splits_offsets = symm_mem.empty(
            (2, nsplits),
            dtype=torch.int64,
            device=self.device,
        ).fill_(-1)

        dist.barrier()

        kraken_out, kraken_splits_offsets = kraken.comm.moe_a2a_dispatch(
            kraken_inp,
            kraken_splits,
            max_out_tokens=max_out_tokens,
            major_align=align,
        )
        torch.ops.symm_mem.all_to_all_vdev_2d(
            ref_inp,
            ref_out,
            ref_splits,
            ref_splits_offsets,
            dist.group.WORLD.group_name,
            major_align=align,
        )

        self._assert_segmented_close(
            kraken_out,
            kraken_splits_offsets,
            ref_out,
            ref_splits_offsets,
        )

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_combine_matches_nvshmem_vdev_2d_offset(self) -> None:
        self._init_process()

        ne = 4
        hidden = 16
        k = 6
        nsplits = ne * self.world_size
        splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=self.device)
        offsets = torch.arange(0, k * nsplits, k, dtype=torch.int64, device=self.device)
        max_in_tokens = k * nsplits
        max_out_tokens = max_in_tokens * self.world_size

        kraken_inp = self._make_input(max_in_tokens, hidden, torch.bfloat16)
        ref_inp = self._make_input(max_in_tokens, hidden, torch.bfloat16)
        kraken_splits_offsets = symm_mem.empty(
            (2, nsplits),
            dtype=torch.int64,
            device=self.device,
        )
        ref_splits_offsets = symm_mem.empty(
            (2, nsplits),
            dtype=torch.int64,
            device=self.device,
        )
        kraken_splits_offsets[0].copy_(splits)
        kraken_splits_offsets[1].copy_(offsets)
        ref_splits_offsets[0].copy_(splits)
        ref_splits_offsets[1].copy_(offsets)

        ref_out = symm_mem.empty(
            (max_out_tokens, hidden),
            dtype=torch.bfloat16,
            device=self.device,
        )
        ref_out_splits_offsets = symm_mem.empty(
            (2, nsplits),
            dtype=torch.int64,
            device=self.device,
        ).fill_(-1)

        dist.barrier()

        kraken_out, kraken_out_splits_offsets = kraken.comm.moe_a2a_combine(
            kraken_inp,
            kraken_splits_offsets,
            max_out_tokens=max_out_tokens,
        )
        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            ref_inp,
            ref_out,
            ref_splits_offsets,
            ref_out_splits_offsets,
            dist.group.WORLD.group_name,
        )

        self._assert_segmented_close(
            kraken_out,
            kraken_out_splits_offsets,
            ref_out,
            ref_out_splits_offsets,
        )

        dist.barrier()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(2)
    def test_dispatch_then_combine_round_trip(self) -> None:
        self._init_process()

        ne = 4
        hidden = 16
        k = 6
        align = 8
        nsplits = ne * self.world_size
        splits = torch.randint(k, (nsplits,), dtype=torch.int64, device=self.device)
        max_in_tokens = k * nsplits
        max_dispatch_tokens = max_in_tokens * self.world_size + ne * align

        inp = self._make_input(max_in_tokens, hidden, torch.float32)
        in_splits = self._symm_copy(splits)

        dist.barrier()

        dispatch_out, dispatch_splits_offsets = kraken.comm.moe_a2a_dispatch(
            inp,
            in_splits,
            max_out_tokens=max_dispatch_tokens,
            major_align=align,
        )
        combine_out, combine_splits_offsets = kraken.comm.moe_a2a_combine(
            dispatch_out,
            dispatch_splits_offsets,
            max_out_tokens=max_in_tokens,
        )

        input_tokens = in_splits.sum().item()
        torch.testing.assert_close(
            combine_out[:input_tokens],
            inp[:input_tokens],
        )
        torch.testing.assert_close(combine_splits_offsets[0], in_splits)
        expected_offsets = torch.cumsum(in_splits, dim=0) - in_splits
        torch.testing.assert_close(combine_splits_offsets[1], expected_offsets)

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    run_tests()
