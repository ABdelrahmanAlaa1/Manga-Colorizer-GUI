import torch

from denoising.denoiser import FFDNetDenoiser


class MangaDenoiser:
    def __init__(self, config):
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("[-] CUDA not available, using CPU.")
            self.device = 'cpu'
        else:
            self.device = config.device

        max_denoise_side = getattr(config, 'denoise_max_side', None)
        if max_denoise_side in (0, None):
            max_denoise_side = None
        self.model = FFDNetDenoiser(self.device, max_side=max_denoise_side)



    def denoise(self, image, sigma=25):
        with torch.no_grad():
            return self.model.get_denoised_image(image, sigma=sigma)
