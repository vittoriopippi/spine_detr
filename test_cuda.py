import torch
import os

def test_cuda():
    print(f'CUDA VERSION:         {torch.version.cuda}')
    print(f'CUDA IS AVAILABLE:    {torch.cuda.is_available()}')   # --> False
    print(f'CUDA DEVICE COUNT:    {torch.cuda.device_count()}')   # --> 0
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        print(f'CUDA VISIBLE DEVICES: {os.environ["CUDA_VISIBLE_DEVICES"]}')
    else: 
        print(f'CUDA VISIBLE DEVICES: {None}')
    if torch.cuda.is_available() and "CUDA_VISIBLE_DEVICES" in os.environ:
        for i in range(torch.cuda.device_count()):
            print(f'\t- {torch.cuda.get_device_name(i)}')
    print('#' * 32 + '\n')

if __name__ == '__main__':
    test_cuda()