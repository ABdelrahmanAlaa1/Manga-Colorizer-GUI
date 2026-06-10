import torch
import os

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

        weights_dir = getattr(config, '_denoiser_weights_dir', None)
        if not weights_dir:
            weights_dir = getattr(config, 'denoiser_weights_dir', None)
        if not weights_dir:
            colorizer_path = getattr(config, 'colorizer_path', '')
            inferred_dir = os.path.dirname(os.path.abspath(colorizer_path)) if colorizer_path else ''
            if inferred_dir and os.path.isdir(inferred_dir):
                weights_dir = inferred_dir
            else:
                weights_dir = os.path.join(os.path.dirname(__file__), 'networks')

        self.model = FFDNetDenoiser(
            self.device,
            max_side=max_denoise_side,
            _weights_dir=os.path.abspath(weights_dir),
            precision_config=config,
        )
        summary = getattr(self.model, 'precision_summary', None)
        self._precision_summary = summary() if callable(summary) else ''



    def denoise(self, image, sigma=25, image_name=None):
        with torch.inference_mode():
            return self.model.get_denoised_image(image, sigma=sigma, image_name=image_name)

    def precision_summary(self):
        summary = getattr(self.model, 'precision_summary', None)
        if callable(summary):
            return summary()
        return self._precision_summary
