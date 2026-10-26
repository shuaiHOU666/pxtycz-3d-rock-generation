# Data layout

Large files (volumes, checkpoints) are **not** stored in git. They are archived at
**<ZENODO-DOI-TO-BE-ADDED>** (see the Open Research statement of the paper).

```
<root>/resolution_<R>/
    <name>+<i>+<j>.npy         uint8 volume (R,R,R); 1 = pore, 0 = solid. <name> is the parent specimen
                               (P01..P06 carbonate, SW01..SW10 sandstone); i, j are the node/edge counts
                               or crop indices used in the file name
    combined_results.txt       whitespace separated; one row per parent specimen: name, Porosity, descriptors,
                               fc1/fc2/fc3 (PxTyCz coordinates), grid_classification (e.g. P1T2C2)
    text_data.txt              training prompts (scripts/make_text_prompts.py)
```

`data/example/combined_results.txt` is the descriptor / PxTyCz table of the 16 specimens used in the paper;
`data/example/measured_properties_original_specimens.txt` lists the simulated bulk modulus, shear modulus and
permeability of the 16 original sub-volumes.

## Voxel sizes
Carbonate specimens (P01–P06) were imaged at 8 µm/voxel, sandstone specimens (SW01–SW10) at 6 µm/voxel
(from the file names of the segmented volumes). Sub-volumes are cropped to 128³ voxels.
> TODO (authors): state here whether the 128³ volumes were down-sampled and the resulting physical voxel size.

## Checkpoints
`df_steps-*.pth` files contain the VQ-VAE, the UNet, the (frozen) SciBERT and the fixed projection layer.
A checkpoint can only be loaded with a VQ-VAE config whose `in_channels/out_ch`, `ch`, `ch_mult` match the one used
for training (`configs/vqvae_*.yaml`).
