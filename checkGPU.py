import torch

if torch.cuda.is_available():
    print("目前 GPU 代號: " + str(torch.cuda.current_device()))
else:
    print("不支援 GPU")
