## 라이브러리 추가하기
import argparse

import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

from train import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="PGGAN",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num_workers', type=int, default=8, help='')
parser.add_argument("--gpu_devices", type=int, nargs='+', default=None, help="")
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--rank', default=0, type=int, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--distributed', action='store_true', help='')

parser.add_argument("--mode", default="train_single", choices=["train_single","train_multi", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
#parser.add_argument("--num_epoch", default=20000, type=int, dest="num_epoch")
parser.add_argument("--num_epoch", default=2000000000000000000, type=int, dest="num_epoch")

parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")


parser.add_argument("--task", default="cyclegan", choices=['cyclegan'], type=str, dest="task")
parser.add_argument('--opts', nargs='+', default=['direction', 0], dest='opts')

parser.add_argument("--ny", default=256, type=int, dest="ny")
parser.add_argument("--nx", default=256, type=int, dest="nx")
parser.add_argument("--nch", default=3, type=int, dest="nch")
parser.add_argument("--nker", default=64, type=int, dest="nker")

parser.add_argument("--wgt_cycle", default=1e1, type=float, dest="wgt_cycle")
parser.add_argument("--wgt_ident", default=5e-1, type=float, dest="wgt_ident")
parser.add_argument("--norm", default='inorm', type=str, dest="norm")

parser.add_argument("--network", default="PGGAN", choices=['DCGAN', 'pix2pix', 'CycleGAN','PGGAN'], type=str, dest="network")
parser.add_argument("--learning_type", default="plain", choices=["plain", "residual"], type=str, dest="learning_type")

args = parser.parse_args()

if __name__ == "__main__":

    if args.mode == "train_single":
        train(0,1,args)

    elif args.mode == "train_multi":
        gpu_devices = ','.join([str(id) for id in args.gpu_devices])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_devices

        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(train, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    elif args.mode == "test":
        test(args)
