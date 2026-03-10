import os
import argparse
import cv2
import numpy as np
import random
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from model.RefineData import RefineData
from model.AIMS import AIMS

def seed_worker(worker_id):
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)

def setup():

    if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--i", help="Input image path")
    device, local_rank = setup()

    model = AIMS().to(device).to(memory_format=torch.channels_last)
    model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
    args = ap.parse_args()
    is_dist = dist.is_available() and dist.is_initialized()
    is_rank0 = (not is_dist) or dist.get_rank() == 0
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, betas=(0.9, 0.99), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, threshold=1e-3, min_lr=1e-4)
        

    dataset = RefineData(
            image_dir=(args.i),
        )
    gt_path = Path(args.i)
    gt_path = Path(gt_path / "sample00.txt")
    with open(gt_path, "r") as f:
        gt_volume = float(f.read().strip())

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,          # crop batch
        shuffle=False,          # dataset이 랜덤 샘플링함
        num_workers=4,          # 4부터 시작해도 됨
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        worker_init_fn=seed_worker,
        drop_last=True
        )
    optimizer.zero_grad(set_to_none=True)
    loss_dict= {}
    for step, sample in enumerate(loader, start=1):
        image  = sample.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        loss_dict = model.module.train_step(step, image, gt_volume, model, optimizer, scheduler)
    if is_rank0:
        print(loss_dict)
if __name__ == "__main__":
    main()