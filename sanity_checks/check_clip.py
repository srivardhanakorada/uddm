import torch

clip_vec = torch.load("data/clip/0_clip.pt")
print(clip_vec.shape)
print(torch.linalg.norm(clip_vec))