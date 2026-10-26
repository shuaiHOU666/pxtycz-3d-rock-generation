"""Training objective of the VQ-VAE: L1 reconstruction + codebook/commitment loss."""
import torch
import torch.nn as nn


class VQLoss(nn.Module):
    def __init__(self, codebook_weight: float = 1.0):
        super().__init__()
        self.codebook_weight = codebook_weight

    def forward(self, codebook_loss, inputs, reconstructions):
        rec = torch.abs(inputs.contiguous() - reconstructions.contiguous()).mean()
        total = rec + self.codebook_weight * codebook_loss.mean()
        log = {
            "loss_total": total.detach(),
            "loss_rec": rec.detach(),
            "loss_codebook": codebook_loss.detach().mean(),
        }
        return total, log
