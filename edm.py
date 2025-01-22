import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from tqdm import tqdm
from functools import partial
from utils.edm_helper import *
from sampling.edm_sampler import edm_sampler
import autoencoder

class EDMPrecond(nn.Module):
    def __init__(self,
                 img_resolution,
                 img_channels,
                 num_classes=0,
                 sigma_min=0,
                 sigma_max=float('inf'),
                 sigma_data=0.5,
                 model=None,
                 ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_classes = num_classes
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = model

    def forward(self, x, sigma, class_labels=None, cfg_scale=None, **model_kwargs):
        model_fn = self.model if cfg_scale is None else partial(self.model.forward_with_cfg, cfg_scale=cfg_scale)
        sigma = sigma.to(x.dtype).reshape(-1, 1, 1, 1)

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        model_out = model_fn(
            x=(c_in * x).to(x.dtype), 
            t=c_noise.flatten(), 
            y=class_labels,
            **model_kwargs
            )
        F_x = model_out['x']
        D_x = c_skip * x + c_out * F_x
        model_out['x'] = D_x
        return model_out

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class EDMDiffusion(nn.Module):
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, sigma_min=0, sigma_max=float('inf'), loss_type='simple'):
        super().__init__()
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        num_gpus = torch.cuda.device_count()
        self.use_distributed = num_gpus > 1
        self.loss_type = loss_type
        self.sampler_fn = edm_sampler
        
    def wrap_model_with_precond(self, model):
        precond = EDMPrecond(img_resolution=model.input_size, img_channels=model.in_channels,
                                num_classes=model.num_classes, sigma_min=self.sigma_min, sigma_max=self.sigma_max,
                                sigma_data=self.sigma_data, model=model)
        return precond

    def forward(self, model, x, sigma, y, cfg_scale=None, **model_kwargs):
        model_out = model(x, sigma, y, cfg_scale=cfg_scale, **model_kwargs)
        return model_out
    
    def get_training_loss(self, net, x, y=None, mask_ratio=0.0, mae_loss_coef=0.0, class_drop_prob=0.1):
        if self.loss_type == "simple":
            return self.get_training_loss_simple(net, x, y, mask_ratio, mae_loss_coef, class_drop_prob)
        else:
            return self.get_training_loss_mae_masking(net, x, y, mask_ratio, mae_loss_coef, class_drop_prob)
    
    def get_training_loss_mae_masking(self, net, x, y=None, mask_ratio=0.0, mae_loss_coef=0.1, class_drop_prob=0.1):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        n = torch.randn_like(x, dtype=x.dtype)
        n_sigma = n * sigma

        model_out = net(x + n_sigma, sigma, y, mask_ratio=mask_ratio, class_drop_prob=class_drop_prob)

        D_yn = model_out['x']
        assert D_yn.shape == x.shape, "Output shape mismatch"

        mse_loss = weight * ((D_yn - x) ** 2)
        
        if mask_ratio > 0:
            assert net.training and 'mask' in model_out, "Mask ratio > 0 but mask not found in model output"

            if hasattr(net, 'patch_size'):
                patch_size = net.patch_size
            elif hasattr(net.module.model, 'patch_size'):
                patch_size = net.module.model.patch_size
            else:
                patch_size = 1

            per_patch_mse_loss = F.avg_pool2d(mse_loss.mean(dim=1), patch_size).flatten(1)

            total_unmasked_loss = torch.zeros(mse_loss.shape[0], device=mse_loss.device) 
            total_mae_loss = torch.zeros(mse_loss.shape[0], device=mse_loss.device)
            num_masks = len(model_out['mask'])

            for mask in model_out['mask']:
                unmask = 1 - mask
                loss_unmask = (per_patch_mse_loss * unmask).sum(dim=1) / unmask.sum(dim=1)
                total_unmasked_loss += loss_unmask

                if mae_loss_coef > 0:
                    mae_loss_value = mae_loss(net.module if self.use_distributed else net, x + n_sigma, D_yn, mask)
                    total_mae_loss += mae_loss_value
                    
            total_unmasked_loss /= num_masks
            if mae_loss_coef > 0:
                total_mae_loss /= num_masks

            loss = total_unmasked_loss
            if mae_loss_coef > 0:
                loss += mae_loss_coef * total_mae_loss
            assert loss.ndim == 1, "Loss should be a 1D tensor"
        else:
            loss = mean_flat(mse_loss)

        raw_net = unwrap_model(net)
        if mask_ratio == 0.0 and raw_net.model.mask_token is not None:
            loss += 0 * torch.sum(raw_net.model.mask_token)
        assert loss.ndim == 1, "Final loss should be a 1D tensor"
        return loss
    
    def get_training_loss_simple(self, net, x, y=None, mask_ratio=0.0, mae_loss_coef=0.0, class_drop_prob=0.1):
        rnd_normal = torch.randn([x.shape[0], 1, 1, 1], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        n = torch.randn_like(x, dtype=x.dtype) * sigma
        model_out = net(x + n, sigma, y, mask_ratio=mask_ratio, class_drop_prob=class_drop_prob)
        D_yn = model_out['x']
        loss = weight * ((D_yn - x) ** 2)
        return loss
    
    @torch.no_grad()
    def generate(self, cfg, net, device, rank, size, outdir):
        seeds = parse_int_list(cfg.seeds)[:cfg.fid_num_samples]
        raw_net = unwrap_model(net)
        in_channels = raw_net.model.in_channels
        input_size = raw_net.model.input_size
        num_classes = raw_net.model.num_classes
        
        
        num_batches = ((len(seeds) - 1) // (cfg.max_batch_size * size) + 1) * size
        all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
        rank_batches = all_batches[rank:: size]

        net.eval()

        sampler_kwargs = dict(num_steps=cfg.num_steps, S_churn=cfg.S_churn,
                            solver=cfg.solver, discretization=cfg.discretization,
                            schedule=cfg.schedule, scaling=cfg.scaling)
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        print(f"sampler_kwargs: {sampler_kwargs}, \nsampler fn: {self.sampler_fn.__name__}")
        vae = autoencoder.get_model(cfg.pretrained_path).to(device)

        num_gpus = torch.cuda.device_count()
        use_distributed = num_gpus > 1
        for batch_seeds in tqdm(rank_batches, unit='batch', disable=(rank != 0)):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, in_channels, input_size, input_size], device=device)
            if num_classes:
                class_labels = rnd.randint(0, num_classes, size=[batch_size], device=device)

            if cfg.class_idx is not None:
                class_labels[:, :] = 0
                class_labels[:, cfg.class_idx] = 1

            feat = None
            
            def recur_decode(z):
                try:
                    return vae.decode(z)
                except:
                    assert z.shape[2] % 2 == 0
                    z1, z2 = z.tensor_split(2)
                    return torch.cat([recur_decode(z1), recur_decode(z2)])
            with torch.no_grad():
                z = self.sampler_fn(net, latents.float(), class_labels.float(), randn_like=rnd.randn_like,
                            cfg_scale=cfg.cfg_scale, feat=feat, **sampler_kwargs).float()
                images = recur_decode(z)
                
            images_np = images.add_(1).mul(127.5).clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for seed, image_np in zip(batch_seeds, images_np):
                image_dir = os.path.join(outdir, f'{seed - seed % 1000:06d}') if cfg.subdirs else outdir
                os.makedirs(image_dir, exist_ok=True)
                image_path = os.path.join(image_dir, f'{seed:06d}.png')
                if image_np.shape[2] == 1:
                    PIL.Image.fromarray(image_np[:, :, 0], 'L').save(image_path)
                else:
                    PIL.Image.fromarray(image_np, 'RGB').save(image_path)               