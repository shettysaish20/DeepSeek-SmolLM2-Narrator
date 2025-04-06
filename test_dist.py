import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    tensor = torch.ones(1).cuda(rank)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank}/{world_size}: Tensor after all-reduce = {tensor.item()}")

if __name__ == '__main__':
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)