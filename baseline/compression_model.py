import torch
from compressai.models import ScaleHyperprior


class VAE_hyperprior(ScaleHyperprior):
    def __init__(self, N, M, **kwargs):
        super().__init__(N, M, **kwargs)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "y_hat": y_hat,
            "z_hat": z_hat,
            "scales_hat": scales_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
