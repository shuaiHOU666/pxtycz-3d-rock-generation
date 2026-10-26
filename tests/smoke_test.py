"""End-to-end smoke test on CPU with tiny networks and synthetic data.

It exercises every stage of the pipeline (VQ-VAE training step, text-conditioned diffusion training
step, DDIM sampling, binarisation, porosity measurement) in well under a minute, without GPU, real
data, pretrained weights or network access (a tiny randomly-initialised BERT replaces SciBERT).

    python tests/smoke_test.py
"""
import os
import sys
import tempfile

import numpy as np
import torch
from omegaconf import OmegaConf
from scipy.ndimage import gaussian_filter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from porefusion.data import to_one_hot  # noqa: E402
from porefusion.diffusion import TextLatentDiffusion  # noqa: E402
from porefusion.prompts import porosity_prompt, topology_prompt, training_prompt  # noqa: E402
from porefusion.vqvae import VQVAETrainer, latent_shape  # noqa: E402

RES = 16


def synthetic_volume(seed: int, phi: float) -> np.ndarray:
    """Gaussian random field thresholded to a target porosity (1 = pore)."""
    rng = np.random.default_rng(seed)
    f = gaussian_filter(rng.standard_normal((RES,) * 3), 1.5)
    return (f > np.quantile(f, 1 - phi)).astype(np.uint8)


def tiny_text_encoder(texts, workdir):
    from transformers import BertConfig, BertModel, BertTokenizerFast
    words = sorted({w for t in texts for w in t.lower().split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + words
    vp = os.path.join(workdir, "vocab.txt")
    open(vp, "w").write("\n".join(vocab))
    tok = BertTokenizerFast(vp, do_lower_case=True)
    model = BertModel(BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
                                 num_attention_heads=2, intermediate_size=64, max_position_embeddings=64))
    return model, tok


def main():
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as wd:
        vq_cfg = os.path.join(wd, "vq.yaml")
        OmegaConf.save(OmegaConf.create({
            "model": {"params": {"embed_dim": 3, "n_embed": 32, "ddconfig": {
                "double_z": False, "z_channels": 3, "resolution": RES, "in_channels": 2, "out_ch": 2,
                "ch": 8, "ch_mult": [1, 2], "num_res_blocks": 1, "attn_resolutions": [], "dropout": 0.0}}},
            "lossconfig": {"params": {"codebook_weight": 1.0}}}), vq_cfg)
        df_cfg = os.path.join(wd, "df.yaml")
        OmegaConf.save(OmegaConf.create({
            "model": {"params": {"linear_start": 0.00085, "linear_end": 0.012, "conditioning_key": "crossattn",
                                 "timesteps": 50, "scale_factor": 0.18215}},
            "unet": {"params": {"image_size": 8, "in_channels": 3, "out_channels": 3, "model_channels": 32,
                                "num_res_blocks": 1, "attention_resolutions": [2], "channel_mult": [1, 2],
                                "num_heads": 2, "dims": 3, "use_spatial_transformer": True,
                                "transformer_depth": 1, "context_dim": 32, "use_checkpoint": False,
                                "legacy": False}}}), df_cfg)

        phis = [0.05, 0.10, 0.15, 0.20]
        vols = [synthetic_volume(i, p) for i, p in enumerate(phis)]
        texts = [training_prompt(f"S{i}", "P1T2C2", f"{p:.4f}") for i, p in enumerate(phis)]
        x = torch.stack([to_one_hot(v) for v in vols])
        assert x.shape == (4, 2, RES, RES, RES)

        # Stage I
        vq = VQVAETrainer(vq_cfg, lr=1e-3, device="cpu")
        losses = [vq.step(x)["loss_total"] for _ in range(5)]
        print("VQ-VAE loss:", [round(l, 3) for l in losses])
        vq_ckpt = os.path.join(wd, "vq.pth")
        vq.save(vq_ckpt)
        assert latent_shape(vq_cfg) == (3, 8, 8, 8)

        # Stage II
        enc = tiny_text_encoder(texts + [topology_prompt("P1T2C2"), porosity_prompt("0.10")], wd)
        model = TextLatentDiffusion(df_cfg, vq_cfg, vq_ckpt, device="cpu", lr=1e-3, text_encoder=enc, max_text_len=32)
        losses = [model.train_step(x, texts)["loss"] for _ in range(3)]
        print("diffusion loss:", [round(l, 3) for l in losses])

        # sampling with partial prompts, as in the paper's evaluation
        for prompt in (topology_prompt("P1T2C2"), porosity_prompt("0.10")):
            logits = model.txt2shape(prompt, ngen=2, ddim_steps=5)
            vol = model.to_binary(logits)
            assert vol.shape == (2, RES, RES, RES) and set(np.unique(vol)) <= {0, 1}
            print(f"{prompt!r}: porosity of samples = {vol.mean(axis=(1, 2, 3)).round(3).tolist()}")

        # checkpoint round trip
        ck = os.path.join(wd, "df.pth")
        model.save(ck)
        model.load(ck)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
