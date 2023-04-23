import os
import math
import torch
import logging
import argparse
import tempfile
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.encoders import PointNet_CLS_Encoder, DGCNN_CLS_Encoder
from utils.train_eval_utils import train_one_epoch, evaluate
from utils.distributed_utils import init_distributed_mode, dist, cleanup
from data.shapenet_loader import ShapeNetCLS
from models.iccv_model import SimAttention_ICCV_1


def init_process(rank, world_size, args):
    # 初始化各进程环境 start
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    args.rank = rank
    args.world_size = world_size
    args.gpu = rank
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    dist.barrier()

    rank = args.rank
    args.lr *= 1  # 学习率要根据并行GPU的数量进行倍增
    checkpoint_path = os.path.join(tempfile.gettempdir(), "initial_weights.pt")
    device = torch.device(args.device)
    return checkpoint_path, rank, device


def get_sampler_and_dataloader(rank, args):
    # 实例化训练数据集
    train_data_set = ShapeNetCLS(args.root, npoint=1024)

    # 给每个rank对应的进程分配训练的样本索引
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data_set)

    # 将样本索引每batch_size个元素组成一个list
    train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)

    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])  # number of workers
    if rank == 0:
        print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_data_set,
                                               batch_sampler=train_batch_sampler,
                                               pin_memory=True,
                                               num_workers=nw)

    return train_sampler, train_loader


def get_model(device, checkpoint_path, rank, args):
    if args.model_choice == 0:
        print('PointNet_CLS_Encoder().cuda(), ')
        online_encoder = PointNet_CLS_Encoder().cuda()
    else:
        print('DGCNN_CLS_Encoder().cuda(), ')
        online_encoder = DGCNN_CLS_Encoder().cuda()

    model = SimAttention_ICCV_1(patch_num=args.patch_num)

    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)
    dist.barrier()

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if args.syncBN:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    return model


def get_optimizer_and_scheduler(model, args):
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-6)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)
    if args.opt is 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    return optimizer, scheduler


def main_fn(rank, world_size, args):
    checkpoint_path, rank, device = init_process(rank, world_size, args)
    train_sampler, train_loader = get_sampler_and_dataloader(rank, args)
    model = get_model(device, checkpoint_path, rank, args)
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    if rank == 0:  # 在第一个进程中打印信息，并实例化tensorboard
        print(args)
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter()
        if os.path.exists("./weights_1017") is False:
            os.makedirs("./weights_1017")

    print('Start training...')
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_mean_loss = train_one_epoch(model=model,
                                          optimizer=optimizer,
                                          data_loader=train_loader,
                                          device=device,
                                          epoch=epoch)
        scheduler.step()

        if rank == 0:
            tags = ["train_loss", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_mean_loss, epoch)
            tb_writer.add_scalar(tags[1], optimizer.param_groups[0]["lr"], epoch)
            torch.save(model.module.state_dict(), "./weights_1017/model_knn_1024_fps-{}-0-v2-new.pth".format(epoch))

    # 删除临时缓存文件
    if rank == 0:
        if os.path.exists(checkpoint_path) is True:
            os.remove(checkpoint_path)
    cleanup()


def run(ws, terminal_parameters):
    processes = []
    for rank in range(ws):
        p = mp.Process(target=main_fn, args=(rank, ws, terminal_parameters))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights_1017')  # weights save path
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--opt', type=str, default='adam')  # or 'sgd' 'adamw'
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--patch_num', type=int, default=8)  # scale 0, 1, 2
    parser.add_argument('--attn_type', type=str, default='cross')  # or 'assgin'

    # 是否启用SyncBatchNorm
    parser.add_argument('--syncBN', type=bool, default=True)
    parser.add_argument('--model_choice', type=int, default=1)  # 1 for DGCNN, 0 for PointNet
    # 数据集所在根目录
    parser.add_argument('--root', type=str,
                        default='/mnt/longvideo/jinkun/4_liang/modelnet40_normal_resampled')

    # 不要改该参数，系统会自动分配
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    # 开启的进程数(注意不是线程),不用设置该参数，会根据nproc_per_node自动设置
    parser.add_argument('--world_size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    opt = parser.parse_args()

    world_size = opt.world_size
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main_fn, args=(rank, world_size, opt))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
