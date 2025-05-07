import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import clip

def generate_clip_triplets(output_folder, mode=False, device="cuda:1"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    aug1 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758)),
    ])
    aug2 = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4815, 0.4578, 0.4082), std=(0.2686, 0.2613, 0.2758)),
    ])
    img_files = sorted([f for f in os.listdir(os.path.join(output_folder, "images")) if f.endswith(".png")])
    os.makedirs(os.path.join(output_folder, "clip_aug"), exist_ok=True)
    os.makedirs(os.path.join(output_folder, "clip"), exist_ok=True)
    for fname in tqdm(img_files, desc="Generating CLIP Embeddings"):
        base = fname.replace(".png", "")
        img_path = os.path.join(output_folder, "images", fname)
        clip_orig_path = os.path.join(output_folder, "clip", f"{base}_clip.pt")
        image = Image.open(img_path).convert("RGB")
        image_orig = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad(): clip_orig = model.encode_image(image_orig).squeeze(0).cpu()
        torch.save(clip_orig, clip_orig_path)
        if mode:
            clip1_path = os.path.join(output_folder, "clip_aug", f"{base}_clip1.pt")
            clip2_path = os.path.join(output_folder, "clip_aug", f"{base}_clip2.pt")
            image1 = aug1(image).unsqueeze(0).to(device)
            image2 = aug2(image).unsqueeze(0).to(device)
            with torch.no_grad():
                clip1 = model.encode_image(image1).squeeze(0).cpu()
                clip2 = model.encode_image(image2).squeeze(0).cpu()
            torch.save(clip1, clip1_path)
            torch.save(clip2, clip2_path)

generate_clip_triplets("/home/teja/vardhan/vlm_assisted_debiasing/uddm/data",mode=True)