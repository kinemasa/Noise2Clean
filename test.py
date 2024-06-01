## pytorch環境でgpuが利用できるかを確認するためのファイル


import torch

## cudaが利用可能か確認する
print(torch.cuda.is_available())
## cudaのデバイス数を確認する
print(torch.cuda.device_count())
## cudaの名前を確認する
print(torch.cuda.get_device_name())




