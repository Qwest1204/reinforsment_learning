import os
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import ray
from ray.train import ScalingConfig
from train_utils import *
from ray.train.torch import TorchTrainer
from ray.train import Checkpoint


parser = argparse.ArgumentParser(description='Train VAE model on HPC RAY cluster')
parser.add_argument('--dataloader', type=str, help='dataloader of data for training (create this on create_dataloader.py)')
parser.add_argument('--zsize', type=int, help='size of z parametr')
#parser.add_argument('--imgsize', type=int, help='size of image')
parser.add_argument('--batchsize', type=int, help='size of batch ')
parser.add_argument('--epoch', type=int, help='how epoch will trained on this data')
parser.add_argument('--checkpoint', type=str, help='previus checkpoint of model')
#parser.add_argument('--raycluster', type=str, help='ray cluster addres')
parser.add_argument('--numworkers', type=int)
parser.add_argument('--usegpu', type=bool)

args = parser.parse_args()

scaling_config = ScalingConfig(num_workers=args.numworkers, use_gpu=args.usegpu)

config = {"s3checkpoint": args.checkpoint, "s3dataloader": args.dataloader, "latent_dim": args.zsize, "batchsize": args.batchsize, "epochs": args.epoch}

trainer = ray.train.torch.TorchTrainer(
    train,
    scaling_config=scaling_config,
    train_loop_config=config,
    # [5a] If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)
result = trainer.fit()

with result.checkpoint.as_directory() as checkpoint_dir:
    #checkpoint = torch.load('/home/qwest/project/PycharmProjects/Reinforsment_Learning/VAE/weights/weights/main/VAE_checkpoint_32_175.pt')
    checkpoint = torch.load(os.path.join(checkpoint_dir, "VAE.pt"))