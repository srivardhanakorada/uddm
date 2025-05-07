import torch

h = torch.load("data/h/0.pt")
print(len(h),h[0].shape)
print(h[0][0,0,:])