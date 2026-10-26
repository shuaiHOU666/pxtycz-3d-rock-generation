# Acknowledgements and third-party code

This repository is a derivative work of

* **MOFFUSION** — J. Park et al., multi-modal conditioning diffusion model for MOF generation,
  <https://github.com/parkjunkil/MOFFUSION>, MIT License, Copyright (c) 2024 parkjunkil.
  The VQ-VAE / latent-diffusion / DDIM / text-conditioning code in `porefusion/` is adapted from it.
  The original licence text is kept in `LICENSE`.

MOFFUSION in turn builds on

* **SDFusion** — <https://github.com/yccyenchicheng/SDFusion>
* **Latent Diffusion Models (LDM)** — <https://github.com/CompVis/latent-diffusion>
* **AutoSDF** — <https://github.com/yccyenchicheng/AutoSDF>
* **taming-transformers** (VQ-VAE encoder/decoder blocks) — <https://github.com/CompVis/taming-transformers>

Please consult those repositories for their own licence terms.

Pore-network and morphology descriptors use **PoreSpy**, **OpenPNM**, **scikit-image** and **SciPy**.
Text embeddings use **SciBERT** (`allenai/scibert_scivocab_uncased`, downloaded from the Hugging Face Hub).
Symbolic regression uses **TorchSISSO**.

## What was changed with respect to MOFFUSION

* data: MOF signed-distance fields were replaced by binary (pore/solid) rock volumes encoded as 2-channel one-hot;
* conditioning: prompts are built from a PxTyCz pore-topology label and the porosity (`porefusion/prompts.py`);
* new modules: PxTyCz topology extraction (`topology/`), best-of-N analysis (`analysis/`), SISSO fitting (`sisso/`);
* training / sampling wrappers were rewritten (`porefusion/vqvae.py`, `porefusion/diffusion.py`); the network
  modules in `porefusion/networks/` are copied with only minor edits (removed debugger calls and prints, device
  handling in the DDIM sampler, optional channel dimension in the VQ-VAE encoder).
