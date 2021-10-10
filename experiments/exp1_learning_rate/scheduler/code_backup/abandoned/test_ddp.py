import sys
import time
import argparse
import os

import re
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import Subset

sys.path.append('../mymodel')
import mymodel.mynet as mymodel
import mymodel.mydataloader as mydataloader

parser = argparse.ArgumentParser()
parser.add_argument('-d','--device', default='0,1,2,3', type=str,
                    help='cuda devices for distributed training')
parser.add_argument('--local_rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('-b',
                    '--batch_size',
                    default=40,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 3200), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
args = parser.parse_args()


args.device = list(map(int, re.findall(r'\d', args.device)))
# local_rank = args.device[args.local_rank]
local_rank = args.local_rank
torch.cuda.empty_cache()
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')
dist.barrier()

print('init local rank: {}, real rank: {}\n'.format(args.local_rank,local_rank))


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt

whole_set = mydataloader.RtDataset(root_dir='../../blender_dataset', dataset_dir='SmallAngle_dataset')
size_set = len(whole_set)
train_size = int(0.8 * size_set)
train_dataset = Subset(whole_set, range(train_size))
test_dataset = Subset(whole_set, range(train_size, size_set))

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
args.nprocs = len(args.device)
print('Current devices count: ',args.nprocs)
args.batch_size = int(args.batch_size / args.nprocs)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,num_workers=12,pin_memory=True, sampler=train_sampler)

model = mymodel.DeepRtResNet(use_reconstruction=False)
model.cuda(local_rank)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],output_device=local_rank)

optimizer = optim.SGD(model.parameters(),lr=1e-4)

criterion = mymodel.loss_Rt()
criterion.cuda(local_rank)

for epoch in range(100):
   train_sampler.set_epoch(epoch)
   l_sum = 0
   start = time.time()
   for batch_idx, (images, target) in enumerate(train_loader):
      last_batch_time = time.time()
      images = images.cuda(local_rank,non_blocking=True)
      target = target.cuda(local_rank,non_blocking=True)
      output = model(images)
      loss = criterion(output, target)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      torch.distributed.barrier()
      reduced_loss = reduce_mean(loss, args.nprocs)/images.shape[0]
      l_sum += reduced_loss.cpu().item()
      if args.local_rank == 0:
         print('Epoch: ',epoch,' Batch idx:',batch_idx,' one batch time:',time.time()-last_batch_time)

   if args.local_rank==0:
      print('epoch:{}'.format(epoch))
      print('Train_loss:',l_sum/(epoch+1),' Time:',time.time()-start)

dist.destroy_process_group()