import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size))#.to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long()#.to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
    
class CTDiffusion:
    def __init__(self, noise_steps=20, beta_start=5e-6, beta_end=1e-2, img_size=64, device=1):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule(type="softmax").to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self, type):
        t = torch.linspace(0, 1, self.noise_steps)
        if type == 'linear':
            beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif type == 'sinusodial':
            beta = torch.sin(t * (numpy.pi / 2)) * (self.beta_end - self.beta_start) + self.beta_start
        elif type == 'softmax':
            beta = torch.sigmoid((t - 0.5) * 10) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise NotImplementedError
        return beta
        
    def save_progressive_noised_image(self, x, t:torch.Tensor):  
        for i in range(0, int(t)):
            it = torch.Tensor([i]).long().to(self.device)
            noise_image = self.noise_images(x, it)[0]
            plt.imsave(f'./results/adding_noise/{i}_noised.jpg', noise_image[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))   # t range [0, 120)

    def sample(self, model, n, img_in=None, t_=None):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            if img_in is not None:
                x = torch.Tensor(img_in).to(self.device)
            if t_ is None:
                t_ = torch.tensor([self.noise_steps - 1]).to(self.device)
            for i in tqdm(reversed(range(1, t_+1)), position=0):  
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)   # no t+1 since t starts from 0
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # if img_in is not None:
                #     if i % 100 == 0 or (i <= 50 and i % 10 == 0):
                #         plt.imsave(f'./results/by_step/step_{i}.jpg', x[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
                #     if i < 20:
                #         plt.imsave(f'./results/by_step/step_{i}.jpg', x[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        model.train()
        return x