# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm

assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae_efficient_relative_pe

from engine_pretrain import train_one_epoch


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='mae_vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--MASTER_ADDR', default='localhost',
                        help='we use teh local host')
    parser.add_argument("--MASTER_PORT",default="10019")
    parser.add_argument("--distributed",action="store_true")
    parser.add_argument("--window_size",default=7,type=int)
    parser.add_argument("--num_window",default=5, type=int)
    parser.add_argument("--ratio_in_window",default=0.7,type=float)
    parser.add_argument("--ratio_out_window",default = 10,type=int)

    return parser


def main(args):
    import submitit
    from pathlib import Path
    # args.distributed=True

    # job_env = submitit.JobEnvironment()

    # args.gpu = 8
    # args.local_rank = -1
    # args.gpu=[0,1,2,3,4,5,6,7]

    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"]="17299"
    
    # os.environ["WORLD_SIZE"] = "8"
    # os.environ["LOCAL_RANK"]="0"
    # os.environ["RANK"]="0"


    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # simple augmentation
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
    print(dataset_train)

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        print("Sampler_train = %s" % str(sampler_train))
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model
    model = models_mae_efficient_relative_pe.__dict__[args.model](norm_pix_loss=args.norm_pix_loss)

    model.to(device)

    

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    print(args.distributed)
    if args.distributed:
        print(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    bandwidth, summary=profile(model)
    print(bandwidth,summary)

    # # following timm: set wd as 0 for bias and norm layers
    # param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    # optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    # print(optimizer)
    # loss_scaler = NativeScaler()

    # misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # print(f"Start training for {args.epochs} epochs")
    # start_time = time.time()
    # for epoch in range(args.start_epoch, args.epochs):
    #     if args.distributed:
    #         data_loader_train.sampler.set_epoch(epoch)
        

    #     # with torch.autograd.detect_anomaly():
    #     train_stats = train_one_epoch(
    #         model, data_loader_train,
    #         optimizer, device, epoch, loss_scaler,
    #         log_writer=log_writer,
    #         args=args
    #     )
    #     if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
    #         misc.save_model(
    #             args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
    #             loss_scaler=loss_scaler, epoch=epoch)

    #     log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
    #                     'epoch': epoch,}

    #     if args.output_dir and misc.is_main_process():
    #         if log_writer is not None:
    #             log_writer.flush()
    #         with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
    #             f.write(json.dumps(log_stats) + "\n")

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print('Training time {}'.format(total_time_str))





from tqdm import tqdm
import torch.autograd.profiler as profiler
import numpy as np
import time
import torch.nn as nn




# @torch.no_grad()
def profile_for_batch_size(model, cfg, batch_size):
    # z = torch.randn(batch_size, G.z_dim, device=cfg.device)
    # c = None

    # window_size=args.window_size, num_window=args.num_window,ratio_in_window=args.ratio_in_window,ratio_out_window=args.ratio_out_window

    model_input = torch.randn(batch_size, 3, 224,224, device=model.device,dtype = torch.float32)
    pos = torch.zeros(batch_size, 196, dtype=torch.bool)

    times = []

    for i in tqdm(range(cfg["num_warmup_iters"]), desc='Warming up'):
        torch.cuda.synchronize()
        fake_img = model(model_input, window_size=7, num_window=1, ratio_in_window=0.7, ratio_out_window=0)
        # y = fake_img[0, 0, 0, 0].item() # sync
        torch.cuda.synchronize()

    time.sleep(1)

    torch.cuda.reset_peak_memory_stats()

    with profiler.profile(record_shapes=True, use_cuda=True) as prof:
        for i in tqdm(range(cfg["num_profile_iters"]), desc='Profiling'):
            torch.cuda.synchronize()
            start_time = time.time()
            with profiler.record_function("forward"):
                fake_img = model(model_input,  window_size=7, num_window=4, ratio_in_window=0.7, ratio_out_window=0)
                # y = fake_img[0, 0, 0, 0].item() # sync
            torch.cuda.synchronize()
            times.append(time.time() - start_time)

    torch.cuda.empty_cache()
    num_imgs_processed = len(times) * batch_size
    total_time_spent = np.sum(times)
    bandwidth = num_imgs_processed / total_time_spent
    summary = prof.key_averages().table(sort_by="cpu_time_total", row_limit=10)

    print(f'[Batch size: {batch_size}] Mean: {np.mean(times):.05f}s/it. Std: {np.std(times):.05f}s')
    print(f'[Batch size: {batch_size}] Imgs/sec: {bandwidth:.03f}')
    print(f'[Batch size: {batch_size}] Max mem: {torch.cuda.max_memory_allocated(model.device) / 2**30:<6.2f} gb')

    return bandwidth, summary




def profile(model):
    train_batchsize = [8, 16, 32, 64]
    test_batchsize = [8, 16, 32, 64, 128,256,512,1024,2048,4096]
    config = {"device":"cuda",
    "model":"beit",
    "num_warmup_iters":10,
    "num_profile_iters":25,
    "batch_sizes":train_batchsize}

    bandwidths = []
    summaries = []

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')

    for batch_size  in  config['batch_sizes']:
        bandwidth, summary = profile_for_batch_size(model, config,batch_size)
        bandwidths.append(bandwidth)
        summaries.append(summary)
    

    best_batch_size_idx = int(np.argmax(bandwidths))
    print(f'------------ Best batch size for {config["model"]} is {config["batch_sizes"][best_batch_size_idx]} ------------')
    print(summaries[best_batch_size_idx])



if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
