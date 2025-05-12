import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

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

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a

def kl_uniform_loss(proj, centroids, temperature=0.005, scale=1000.0):
    sim = F.cosine_similarity(proj.unsqueeze(1), centroids.unsqueeze(0), dim=2)  # [B, K]
    pt = F.softmax(sim / temperature, dim=1)  # sharpened
    uniform = torch.full_like(pt, 1.0 / pt.size(1))
    kl = F.kl_div(pt.log(), uniform, reduction='batchmean')
    return scale * kl, pt  # scaled

def generalized_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et = model(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
    return xs

def generalized_steps_ret_h(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        h_traj_unmod = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')
            et, h_unmod = model.module.forward_ret_h(xt, t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))
            h_traj_unmod.append(h_unmod.detach().cpu())
    return xs, h_traj_unmod

def generalized_steps_ret_clip(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]
    proj_model = HToCLIPJoint(h_dim=8192, t_dim=128, out_dim=512, num_timesteps=50).to('cuda')
    proj_model.load_state_dict(torch.load(kwargs.get("model_path",None)))
    proj_model.eval()
    mid_point = len(seq)//2
    timestep_map = {step: idx for idx, step in enumerate(seq)}
    store_h = None
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = (torch.ones(n) * i).to(x.device)
        next_t = (torch.ones(n) * j).to(x.device)
        at = compute_alpha(b, t.long())
        at_next = compute_alpha(b, next_t.long())
        xt = xs[-1].to('cuda')
        et, h_unmod = model.module.forward_ret_h(xt, t)
        if i == seq[mid_point]: store_h = h_unmod
        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.to('cpu'))
        c1 = (kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt())
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        xs.append(xt_next.to('cpu'))
    h_flat = store_h.view(store_h.size(0), -1)
    t_surrogate = torch.full((n,), timestep_map[seq[mid_point]], dtype=torch.long, device=x.device)
    est_clip = proj_model(h_flat,t_surrogate)
    return xs,est_clip

def mod_generalized_steps_ret_h(x, seq, model, b, **kwargs):
    n = x.size(0)
    seq_next = [-1] + list(seq[:-1])
    x0_preds = []
    xs = [x]

    proj_model = HToCLIPJoint(h_dim=8192, t_dim=128, out_dim=512, num_timesteps=50).to('cuda')
    proj_model.load_state_dict(torch.load(kwargs.get("model_path", None)))
    proj_model.eval()

    timestep_map = {step: idx for idx, step in enumerate(seq)}
    h_un_mod, h_mod = [], []
    kl_per_timestep = []
    diff_stats = []

    temperature = kwargs.get("temperature", 0.07)
    gamma = kwargs.get("gamma", 0.07)
    clip_centroids = torch.from_numpy(np.load(kwargs.get("centroids_path", None))).float().to('cuda')

    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.full((n,), i, dtype=torch.long, device=x.device)
        next_t = torch.full((n,), j, dtype=torch.long, device=x.device)
        t_surrogate = torch.full((n,), timestep_map[i], dtype=torch.long, device=x.device)

        at = compute_alpha(b, t)
        at_next = compute_alpha(b, next_t)

        xt = xs[-1].to('cuda')
        xt.requires_grad = False

        if 700 <= i <= 900:
            with torch.enable_grad():
                # Original
                et_before, h_before, _ = model.module.forward_with_h(xt, t, gamma=0, grad=None)
                x0_before = (xt - et_before * (1 - at).sqrt()) / at.sqrt()
                proj_before = proj_model(h_before.view(h_before.size(0), -1), t_surrogate)
                kl_before, _ = kl_uniform_loss(proj_before, clip_centroids, temperature)

                # Gradient + edited h
                grad = torch.autograd.grad(kl_before, h_before, retain_graph=True)[0]
                et_after, _, h_after = model.module.forward_with_h(xt, t, gamma=gamma, grad=grad)
                x0_after = (xt - et_after * (1 - at).sqrt()) / at.sqrt()
                proj_after = proj_model(h_after.view(h_after.size(0), -1), t_surrogate)
                kl_after, _ = kl_uniform_loss(proj_after, clip_centroids, temperature)

                # Log deltas
                delta_h = (h_after - h_before).norm().item()
                delta_et = (et_after - et_before).norm().item()
                delta_x0 = (x0_after - x0_before).norm().item()

                print(f"Timestep {i} : KL before = {kl_before.item():.4f}, after = {kl_after.item():.4f}")
                print(f"             ‖Δh‖ = {delta_h:.4f}, ‖Δet‖ = {delta_et:.4f}, ‖Δx₀‖ = {delta_x0:.4f}")

                kl_per_timestep.append((i, kl_before.item(), kl_after.item()))
                diff_stats.append((i, delta_h, delta_et, delta_x0))

                # Use edited et
                et = et_after
                h_un_mod.append(h_before)
                h_mod.append(h_after)
        else:
            with torch.no_grad():
                et, _, _ = model.module.forward_with_h(xt, t, gamma=0, grad=None)

        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
        x0_preds.append(x0_t.cpu())

        c1 = kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
        xs.append(xt_next.cpu())

    return xs, h_un_mod, h_mod, kl_per_timestep, diff_stats

def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')
            output = model(x, t.float())
            e = output
            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)
            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds