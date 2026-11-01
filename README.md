# OSHDRNet: What and When to Trust: Observation Support for Multi-Exposure HDR Imaging

OSHDRNet is an observation-support-driven HDR reconstruction model for dynamic multi-exposure scenes. The complete code will be released upon acceptance of the paper.

## Project structure

```
OSHDRNet
├── dataset/          data loading (released upon acceptance)
├── loss/             training objectives (released upon acceptance)
├── lr_scheduler/     learning-rate schedule
├── models/           RGA, DEA and ODC (released upon acceptance)
├── utils/            utilities (released upon acceptance)
├── train.py          two-stage training (released upon acceptance)
└── test.py           evaluation (released upon acceptance)
```

## Datasets

The datasets used in this project are available from:

- [Kalantari's dataset](https://cseweb.ucsd.edu/~viscomp/projects/SIG17HDR/)
- [Tel's dataset](https://drive.google.com/drive/folders/1CtvUxgFRkS56do_Hea2QC7ztzglGfrlB)
- [Hu's dataset](https://github.com/nadir-zeeshan/sensor-realistic-synthetic-data)

The directory structure of Kalantari's, Tel's and Hu's datasets is the same as in [AFUNet](https://github.com/eezkni/AFUNet). An example for Kalantari's dataset:

```
data_root
└── Kal
    ├── Training
    │   ├── 001
    │   │   ├── 262A0898.tif
    │   │   ├── 262A0899.tif
    │   │   ├── 262A0900.tif
    │   │   ├── exposure.txt
    │   │   └── HDRImg.hdr
    │   ├── 002
    │   ├── ...
    │   └── 074
    └── Test
        └── Test-set
            ├── 001
            │   ├── 262A2615.tif
            │   ├── 262A2616.tif
            │   ├── 262A2617.tif
            │   ├── exposure.txt
            │   └── HDRImg.hdr
            ├── 002
            ├── ...
            └── 015
```

## Acknowledgments

We thank the authors of [AFUNet](https://github.com/eezkni/AFUNet) for releasing their code.
