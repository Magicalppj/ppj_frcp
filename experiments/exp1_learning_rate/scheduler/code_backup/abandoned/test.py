import os
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import torch
n = torch.cuda.device_count()
print('device count: ',n)
devices = list(range(n))
torch.cuda.set_device(device=devices[1])
# a = torch.rand(500,500).to(devices[0])
# b = torch.rand(500,500).to(devices[1])
# c = torch.rand(500,500).to(devices[2])
d = torch.rand(1000,1000).cuda()

start_time = time.time()


while time.time() - start_time <20:
    pass