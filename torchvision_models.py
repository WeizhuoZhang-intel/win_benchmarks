import time
import torch
import torchvision.models as models
from utils import AverageMeter, ProgressMeter

batch_size = 4
warmup_iterations = 10
number_iter = 100
model = models.__dict__["resnet50"](pretrained=True)
model = models.__dict__["mobilenet_v3_large"](pretrained=True)

batch_time = AverageMeter('Time', ':6.3f')
progress = ProgressMeter(
        number_iter,
        [batch_time],
        prefix='Test: ')
images = torch.randn(batch_size, 3, 224, 224).contiguous(memory_format=torch.channels_last)

for i in range(number_iter):
    if i >= warmup_iterations:
        end = time.time()
    output = model(images)

    if i >= warmup_iterations:
        batch_time.update(time.time() - end)

    if i % 10 == 0:
        progress.display(i)

latency = batch_time.avg / batch_size * 1000
perf = batch_size / batch_time.avg

print('inference latency %.3f ms'%latency)
print("Throughput: {:.3f} fps".format(perf))