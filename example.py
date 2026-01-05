import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import kraken
import os

# setup distributed process group. 
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(f"cuda:{local_rank}")
dist.init_process_group("nccl")

# Create and initialize a symmetric memory tensor
# See blog: https://dev-discuss.pytorch.org/t/pytorch-symmetricmemory-harnessing-nvlink-programmability-with-ease/279 for symmetric memory details. 
a_shared = symm_mem.empty(
        (4096, 4096), 
        dtype=torch.bfloat16, 
        device=f"cuda:{local_rank}",
    )
symm_mem.rendezvous(a_shared, group=dist.group.WORLD)
a_shared = a_shared.normal_()

# Call one_shot_all_reduce kernel from kraken. 
a = kraken.comm.one_shot_all_reduce(a_shared)

# all ranks must see the same result after all-reduce
print(f"Rank {dist.get_rank()} completed all-reduce with tensor max, min, mean: {a.max()}, {a.min()}, {a.mean():.4f}")


