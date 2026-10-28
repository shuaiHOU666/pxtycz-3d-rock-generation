# Topology-constrained generation of 3D porous media (PxTyCz + latent diffusion)

Code accompanying the manuscript *Topology-constrained generation of 3D porous media from small data and
interpretable property equations* (submitted to *JGR: Machine Learning and Computation*).
<!-- TODO: add the final title, DOI and Zenodo DOI -->

The workflow has four parts:

1. **Pore topology descriptor (PxTyCz)** – pore-network / morphology descriptors are extracted from segmented micro-CT
   volumes and mapped to a discrete topology label (`topology/`).
2. **Text prompts** – each volume gets a sentence such as
   `P01+1095+1106 exhibits a P1T2C2 topology and a porosity of 0.1448` (`scripts/make_text_prompts.py`).
3. **Generation** – a 3D VQ-VAE and a text-conditioned latent diffusion model (SciBERT, cross-attention,
   DDIM sampling) generate volumes from prompts (`porefusion/`, `scripts/`).
4. **Property equations** – SISSO symbolic regression maps descriptors to bulk/shear modulus and permeability (`sisso/`).

The generative part is derived from [MOFFUSION](https://github.com/parkjunkil/MOFFUSION) (MIT), see `NOTICE.md`.

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/smoke_test.py        # CPU, synthetic data, tiny networks, no downloads, < 1 min
```
The smoke test trains the VQ-VAE and the diffusion model for a few steps on synthetic volumes, samples with
partial prompts and round-trips a checkpoint. It uses a tiny random BERT instead of SciBERT so it needs no weights.

## Reproducing the pipeline
```bash
# 0. data: see docs/DATA.md (archived on Zenodo)

# 1. descriptors and PxTyCz label (run from topology/; paths are set at the bottom of each script)
python topology/crop_and_extract_network_features.py     # crops sub-volumes -> summary.txt, topology.txt
python topology/pxtycz_pca_cluster.py                    # fits PCA/min-max model -> pore_topology_model.pkl/json
python topology/pxtycz_grid_classification.py            # applies the model -> grid_classification_results.csv
python topology/merge_descriptors_and_classes.py         # -> combined_summary_merged.txt (combined_results.txt)

# 2. prompts
python scripts/make_text_prompts.py --root data --res 128

# 3. training (GPU)
python scripts/train_vqvae.py     --root data --res 128 --cfg configs/vqvae_128.yaml --out runs/vq128
python scripts/train_diffusion.py --root data --res 128 --vq-cfg configs/vqvae_128.yaml \
       --vq-ckpt runs/vq128/vqvae_steps-5000.pth --out runs/df128

# 4. generation: N samples per specimen for the prompts "with a <PxTyCz> topology" / "with a porosity of <phi>"
python scripts/generate_batch.py --ckpt runs/df128/df_steps-5000.pth --vq-cfg configs/vqvae_128.yaml \
       --table data/example/combined_results.txt --n-samples 80 --out samples

# 5. evaluation (see "Reporting protocol")
python analysis/select_best_of_n.py --samples samples --out analysis_out

# 6. property equations
python sisso/fit_sisso.py --labels label.txt --descriptors combined_summary_merged.txt --out sisso_results
```
`configs/vqvae_64.yaml` (64³ → 16³ latent) and `configs/vqvae_128.yaml` (128³ → 8³ latent) are both provided;
**use the config that matches your checkpoint**. Pretrained checkpoints: see `docs/DATA.md`.

## Reporting protocol: best-of-N
`scripts/generate_batch.py` stores **all** N samples and logs their measured porosity.
`analysis/select_best_of_n.py` reports (i) statistics over **all** samples and (ii) the best-of-N selection
(the samples closest to the target porosity). The latter uses the target value and is an upper bound, not a
random draw: figures or tables built from it must say "best-of-N" and give N. The generation prompts differ from
the training prompts (partial prompts), and targets are taken from the specimens in the training set.

## Implementation notes
* The SciBERT encoder is frozen; the 768×768 projection (`cond_layer`) is saved in checkpoints but, as in the
  reported experiments, is **not optimised** (a fixed random projection). `TextLatentDiffusion(train_cond_layer=True)` changes this.
* The diffusion model operates on the continuous (pre-quantisation) VQ-VAE latent; the codebook is used for
  training the VQ-VAE only.
* `sisso/fit_sisso.py` scales features with Min-Max on the full table before the random 90/10 split (as in the
  original experiments; `--scale-on-train-only` avoids this). The split is at the sample level.
* Scripts in `topology/` are research code ported from the original working directory: comments and some log
  messages are in Chinese and input/output paths are set in their `__main__` blocks.

## Citation / licence
MIT licence (`LICENSE`), incorporating MOFFUSION's（https://github.com/parkjunkil/MOFFUSION） licence text.
