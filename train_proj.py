import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import umap
import clip
from PIL import Image

class HToCLIPJoint(nn.Module):
    def __init__(self, h_dim=8192, t_dim=128, out_dim=512, num_timesteps=50):
        super().__init__()
        self.time_embed = nn.Embedding(num_timesteps, t_dim)
        self.net = nn.Sequential(
            nn.Linear(h_dim + t_dim, 4096),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048, out_dim)
        )

    def forward(self, h, t):
        t_embed = self.time_embed(t)
        x = torch.cat([h, t_embed], dim=-1)
        x = self.net(x)
        return F.normalize(x, dim=-1)

class TripletDataset(Dataset):
    def __init__(self, base_dir, clip_model, clip_preprocess, device):
        self.h_dir = os.path.join(base_dir, "h")
        self.clip1_dir = os.path.join(base_dir, "clip_aug")
        self.clip2_dir = os.path.join(base_dir, "clip_aug")
        self.img_dir = os.path.join(base_dir, "images")
        self.h_vecs, self.t_vecs, self.clip1_vecs, self.clip2_vecs = [], [], [], []

        valid_ids = [f.replace(".pt", "") for f in os.listdir(self.h_dir) if f.endswith(".pt")]
        for id in tqdm(valid_ids):
            h_path = os.path.join(self.h_dir, f"{id}.pt")
            clip1_path = os.path.join(self.clip1_dir, f"{id}_clip1.pt")
            clip2_path = os.path.join(self.clip2_dir, f"{id}_clip2.pt")
            img_path = os.path.join(self.img_dir, f"{id}.png")

            if not os.path.exists(clip1_path) or not os.path.exists(clip2_path) or not os.path.exists(img_path):
                continue

            try:
                h_list = torch.load(h_path)
                clip1 = F.normalize(torch.load(clip1_path).squeeze().float(), dim=-1)
                clip2 = F.normalize(torch.load(clip2_path).squeeze().float(), dim=-1)
            except:
                continue

            for t, h in enumerate(h_list):
                self.h_vecs.append(h.flatten())
                self.t_vecs.append(t)
                self.clip1_vecs.append(clip1)
                self.clip2_vecs.append(clip2)

    def __len__(self):
        return len(self.h_vecs)

    def __getitem__(self, idx):
        return self.h_vecs[idx], self.t_vecs[idx], self.clip1_vecs[idx], self.clip2_vecs[idx]

def nt_xent_loss(z1, z2, temperature=0.07):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
    labels = torch.arange(B, device=z.device)
    labels = torch.cat([labels, labels], dim=0)
    mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, -9e15)
    positives = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, positives)
    return loss

def compute_recall_at_k(preds, targets, k):
    preds = F.normalize(preds, dim=-1).cpu().float()
    targets = F.normalize(targets, dim=-1).cpu().float()
    sim_matrix = torch.matmul(preds, targets.T)
    topk = torch.topk(sim_matrix, k=k, dim=1).indices
    labels = torch.arange(len(targets))
    hits = [(labels[i] in topk[i]) for i in range(len(targets))]
    return np.mean(hits)

def umap_plot(true_vecs, pred_vecs, save_dir, label_t):
    os.makedirs(save_dir, exist_ok=True)
    reducer = umap.UMAP(n_components=2, random_state=42)
    true_2d = reducer.fit_transform(true_vecs)
    pred_2d = reducer.fit_transform(pred_vecs)

    for name, data_2d in [("true", true_2d), ("pred", pred_2d)]:
        plt.figure()
        plt.scatter(data_2d[:, 0], data_2d[:, 1], s=3)
        plt.title(f"UMAP of {'True' if name == 'true' else 'Predicted'} CLIP Embeddings (t={label_t})")
        plt.savefig(os.path.join(save_dir, f"umap_t{label_t}_{name}.png"))
        plt.close()

def train_htoclip(data_dir, output_dir, batch_size=512, epochs=10, lr=1e-4):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    dataset = TripletDataset(data_dir, clip_model, clip_preprocess, device)
    total = len(dataset)
    train_len = int(0.6 * total)
    val_len = int(0.2 * total)
    test_len = total - train_len - val_len
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    model = HToCLIPJoint().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_log, val_sim_log = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        for h, t, clip1, _, in tqdm(train_loader):
            h, t, clip1 = h.to(device), t.to(device), clip1.to(device)
            z1 = model(h, t)
            loss = nt_xent_loss(z1, clip1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(train_loader)
        loss_log.append(avg_loss)

        model.eval()
        cosine_scores = []
        val_preds, val_trues = [], []
        with torch.no_grad():
            for h, t, _, clip2 in val_loader:
                h, t, clip2 = h.to(device), t.to(device), clip2.to(device)
                pred = model(h, t)
                sim = F.cosine_similarity(pred, clip2, dim=1)
                cosine_scores.extend(sim.cpu().tolist())

                mask = (t == 25)
                if mask.sum() > 0:
                    val_preds.append(pred[mask].cpu())
                    val_trues.append(clip2[mask].cpu())

        avg_sim = sum(cosine_scores) / len(cosine_scores)
        val_sim_log.append(avg_sim)

        recall_1 = compute_recall_at_k(torch.cat(val_preds), torch.cat(val_trues), k=1) if val_preds else 0
        recall_5 = compute_recall_at_k(torch.cat(val_preds), torch.cat(val_trues), k=5) if val_preds else 0

        print(f"Epoch {epoch} | Train Loss: {avg_loss:.4f} | Val Cosine Sim: {avg_sim:.4f} | R@1: {recall_1:.3f} | R@5: {recall_5:.3f}")
        if epoch % 2 == 0 and val_preds:
            umap_plot(
                torch.cat(val_trues).numpy(),
                torch.cat(val_preds).numpy(),
                os.path.join(output_dir, f"val_umap_epoch{epoch}"),
                25
            )
            torch.save(model.state_dict(), os.path.join(output_dir, f"htoclip_joint_epoch{epoch}.pt"))

    torch.save(model.state_dict(), os.path.join(output_dir, "htoclip_joint_final.pt"))

    plt.figure()
    plt.plot(range(1, epochs + 1), loss_log)
    plt.xlabel("Epoch")
    plt.ylabel("NT-Xent Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs + 1), val_sim_log)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Cosine Similarity")
    plt.title("Validation Cosine Similarity")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "val_cosine_sim.png"))
    plt.close()

    model.eval()
    for t_target in range(21, 26):
        pred_vecs, true_vecs = [], []
        with torch.no_grad():
            for h, t, _, clip2 in tqdm(test_loader, desc=f"Testing (t={t_target})"):
                mask = (t == t_target)
                if mask.sum() == 0:
                    continue
                h_masked, t_masked = h[mask].to(device), t[mask].to(device)
                clip2_masked = clip2[mask].to(device)
                pred = model(h_masked, t_masked).cpu()
                pred_vecs.append(pred)
                true_vecs.append(clip2_masked.cpu())
        if pred_vecs:
            pred_all = torch.cat(pred_vecs)
            true_all = torch.cat(true_vecs)
            r1 = compute_recall_at_k(pred_all, true_all, k=1)
            r5 = compute_recall_at_k(pred_all, true_all, k=5)
            print(f"[TEST] t={t_target} | R@1: {r1:.3f} | R@5: {r5:.3f}")
            umap_plot(true_all.numpy(), pred_all.numpy(),
                      os.path.join(output_dir, f"test_umap_t{t_target}"), t_target)

# Run training
train_htoclip(
    data_dir="/home/teja/vardhan/vlm_assisted_debiasing/uddm/data",
    output_dir="/home/teja/vardhan/vlm_assisted_debiasing/uddm/checkpoints"
)
