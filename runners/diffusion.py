import os
import glob
import numpy as np
import tqdm
import torch
from models.diffusion import Model
from models.ema import EMAHelper
from datasets import inverse_data_transform
from functions.denoising import generalized_steps,generalized_steps_ret_h
import torchvision.utils as tvu

def torch2hwcuint8(x, clip=False):
    if clip: x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

class Diffusion(object):

    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None: device = (torch.device("cuda")if torch.cuda.is_available()else torch.device("cpu"))
        self.device = device
        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(beta_schedule=config.diffusion.beta_schedule,beta_start=config.diffusion.beta_start,beta_end=config.diffusion.beta_end,num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,)
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0)
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        if self.model_var_type == "fixedlarge": self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall": self.logvar = posterior_variance.clamp(min=1e-20).log()

    def sample(self,mode):
        model = Model(self.config)
        print("Loading CelebA pretrained checkpoint from pretrained/ckpt.pth")
        states = torch.load("pretrained/ckpt.pth", map_location=self.device)
        model = model.to(self.device)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(model)
        model.eval()
        self.sample_fid(model,mode)

    def sample_fid(self, model, mode):
        config = self.config
        img_id = len(glob.glob(f"{self.args.output}/images/*"))
        print(f"Starting from image {img_id}")
        total_n_samples = getattr(self.config.sampling, "n_samples", 100)
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size
        os.makedirs(os.path.join(self.args.output,"images"),exist_ok=True)
        os.makedirs(os.path.join(self.args.output,"h"),exist_ok=True)
        os.makedirs(os.path.join(self.args.output,"clip"),exist_ok=True)
        os.makedirs(os.path.join(self.args.output,"clip_aug"),exist_ok=True)
        with torch.no_grad():
            for _ in tqdm.tqdm(range(n_rounds), desc="Generating"):
                n = config.sampling.batch_size
                x = torch.randn(n,config.data.channels,config.data.image_size,config.data.image_size,device=self.device,)
                if mode == 0: 
                    if not self.args.collect: 
                        x = self.sample_image(x, model,ret_h = False)
                        x = inverse_data_transform(config, x)
                        for i in range(n):
                            tvu.save_image(x[i], os.path.join(self.args.output, "images" ,f"{img_id}.png"))
                            img_id += 1
                    else: 
                        x,h_traj = self.sample_image(x, model,ret_h = True)
                        x = inverse_data_transform(config, x)
                        for i in range(n):
                            tvu.save_image(x[i], os.path.join(self.args.output, "images" ,f"{img_id}.png"))
                            h_path = os.path.join(self.args.output, "h" ,f"{img_id}.pt")
                            h_i = [h[i].clone() for h in h_traj]
                            torch.save(h_i, h_path)
                            img_id += 1
                else:
                    if not self.args.collect: 
                        x = self.sample_image_debias(x, model,ret_h = False)
                        x = inverse_data_transform(config, x)
                        for i in range(n):
                            tvu.save_image(x[i], os.path.join(self.args.output, "images" ,f"{img_id}.png"))
                            img_id += 1
                    else: 
                        x,h_traj = self.sample_image_debias(x, model,ret_h = True)
                        x = inverse_data_transform(config, x)
                        for i in range(n):
                            tvu.save_image(x[i], os.path.join(self.args.output, "images" ,f"{img_id}.png"))
                            h_path = os.path.join(self.args.output, "h" ,f"{img_id}.pt")
                            h_i = [h[i].clone() for h in h_traj]
                            torch.save(h_i, h_path)
                            img_id += 1

    def sample_image(self, x, model,ret_h=False):
        try: skip = self.args.skip
        except Exception: skip = 1
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        if not ret_h: 
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            return xs[-1]
        else:
            xs,h = generalized_steps_ret_h(x, seq, model, self.betas, eta=self.args.eta)
            return xs[-1],h
        
    ## TODO : Rewrite
    def sample_image_debias(self,x, model, ret_h=False):
        try: skip = self.args.skip
        except Exception: skip = 1
        skip = self.num_timesteps // self.args.timesteps
        seq = range(0, self.num_timesteps, skip)
        if not ret_h: 
            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            return xs[-1]
        else:
            xs,h = generalized_steps_ret_h(x, seq, model, self.betas, eta=self.args.eta)
            return xs[-1],h