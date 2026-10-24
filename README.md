# OSHDRNet: What and When to Trust: Observation Support for Multi-Exposure HDR Imaging

OSHDRNet is an observation-support-driven HDR reconstruction model for dynamic multi-exposure scenes. The complete code will be released upon acceptance of the paper.

## Environment setup

```sh
conda create -n oshdrnet python=3.10
conda activate oshdrnet
pip install torch torchvision
pip install numpy opencv-python einops imageio tqdm tensorboard scikit-image
```

## Data preparation

The datasets used in this project are available from:

- [Kalantari's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
- [Tel's dataset](https://drive.google.com/drive/folders/1CtvUxgFRkS56do_Hea2QC7ztzglGfrlB)
- [Hu's dataset](https://github.com/nadir-zeeshan/sensor-realistic-synthetic-data)

Example structure:

```
data_root
└── Kal
    ├── Training
    │   ├── 001
    │   │   ├── 262A0898.tif
    │   │   ├── 262A0899.tif
    │   │   ├── 262A0900.tif
    │   │   ├── exposure.txt
    │   │   ├── HDRImg.hdr
    │   ├── 002
    │   └── ...
    └── Test
        ├── 001
        │   ├── 262A2615.tif
        │   ├── 262A2616.tif
        │   ├── 262A2617.tif
        │   ├── exposure.txt
        │   ├── HDRImg.hdr
        └── ...
```

By default the scripts read the dataset from `../data/Kal`. Tel's dataset uses the same file names; for Hu's
dataset, set `--exposure_file_name input_exp.txt` and `--label_file_name ref_hdr_aligned_linear.hdr`.

## Running the model

### Stage 1 Training (RGA and DEA)

```sh
python train.py
```

### Stage 2 Training (ODC)

```sh
python train_odc.py
```

### Testing

`test.py` loads the final model (last epoch of stage 2) and runs evaluation; `--save_results` additionally saves the
HDR results, tone-mapped previews and completion regions.

```sh
python test.py
```

## Acknowledgments

We thank the authors of [AFUNet](https://github.com/eezkni/AFUNet) for releasing their code.
