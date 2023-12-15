# CE-OST: Contour Emphasis for One-Stage Transformer-based Camouflage Instance Segmentation

This repository is the official implementation of the paper entitled: **CE-OST: Contour Emphasis for One-Stage Transformer-based Camouflage Instance Segmentation**, presented at [MAPR2023](https://mapr.uit.edu.vn). <br>
**Authors**: Thanh-Danh Nguyen, Duc-Tuan Luu, Vinh-Tiep Nguyen*, and Thanh Duc Ngo.

[[Paper]](https://ieeexplore.ieee.org/document/10288682)


## 1. Environment Setup
Download and install Anaconda with the recommended version from [Anaconda Homepage](https://www.anaconda.com/download): [Anaconda3-2019.03-Linux-x86_64.sh](https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh) 
 
```
git clone https://github.com/danhntd/CEOST.git
cd CEOST
curl -O https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
bash Anaconda3-2019.03-Linux-x86_64.sh
```

After completing the installation, please create and initiate the workspace with the specific versions below. The experiments were conducted on a Linux server with a single `GeForce RTX 2080Ti GPU`, CUDA 11.1, Torch 1.9.

```
conda create --name CEOST python=3
conda activate CEOST
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
```

This source code is based on [Detectron2](https://github.com/facebookresearch/detectron2). Please refer to INSTALL.md for the pre-built or building Detectron2 from source.

After setting up the dependencies, use the command `python setup.py build develop` in this root to finish.

In case you face some environmental conflicts, these installations may help:
```
pip install mxnet-mkl==1.6.0 numpy==1.23.1
pip install protobuf==3.20.* #demo.py
```

## 2. Data Preparation
In this work, we utilize three common datasets for camouflage research, i.e. [COD10K](https://dengpingfan.github.io/pages/COD.html), [NC4K](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment), and [CAMO++](https://sites.google.com/view/ltnghia/research/camo_plus_plus?authuser=0).

### Download the datasets

| Dataset | COD10K | NC4K | CAMO++ |
| ------- |:------:|:----:|:------:|
| Image | [link](https://drive.google.com/file/d/1YGa3v-MiXy-3MMJDkidLXPt0KQwygt-Z/view?usp=sharing) | [link](https://drive.google.com/file/d/1eK_oi-N4Rmo6IIxUNbYHBiNWuDDLGr_k/view?usp=sharing) | [link](https://sites.google.com/view/ltnghia/research/camo_plus_plus?authuser=0#h.z7hqek2t1ln2)     |
| Annotation | [link](https://drive.google.com/drive/folders/1Yvz63C8c7LOHFRgm06viUM9XupARRPif?usp=sharing) | [link](https://drive.google.com/drive/folders/1LyK7tl2QVZBFiNaWI_n0ZVa0QiwF2B8e?usp=sharing) | [link](https://sites.google.com/view/ltnghia/research/camo_plus_plus?authuser=0#h.z7hqek2t1ln2)      |

### Register datasets
Detectron2 requires a step of data registration for those who want to use the external datasets ([Detectron2 Docs](https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html)).

Please modify these lines in `./adet/data/datasets/cis.py` corresponding to your local paths:
```
# change the paths 
COD10K_ROOT = 'COD10K-v3'
COD10K_ANN_ROOT = os.path.join(COD10K_ROOT, 'annotations')
COD10K_TRAIN_PATH = os.path.join(COD10K_ROOT, 'Train/Image')
COD10K_TEST_PATH = os.path.join(COD10K_ROOT, 'Test/Image')
COD10K_TRAIN_JSON = os.path.join(COD10K_ANN_ROOT, 'train_instance.json')
COD10K_TEST_JSON = os.path.join(COD10K_ANN_ROOT, 'test2026.json')

NC4K_ROOT = 'NC4K'
NC4K_PATH = os.path.join(NC4K_ROOT, 'Imgs')
NC4K_JSON = os.path.join(NC4K_ROOT, 'nc4k_test.json')

CAMOPP_ROOT = 'camopp'
CAMOPP_ANN_ROOT = os.path.join(CAMOPP_ROOT, 'Annotations')
CAMOPP_TRAIN_PATH = os.path.join(CAMOPP_ROOT, 'Publish/Image/Train')
CAMOPP_TEST_PATH = os.path.join(CAMOPP_ROOT, 'Publish/Image/Test')
CAMOPP_TRAIN_JSON = os.path.join(CAMOPP_ANN_ROOT, 'train.json')
CAMOPP_TEST_JSON = os.path.join(CAMOPP_ANN_ROOT, 'test.json')
```

## 3. Training Pipeline
Our proposed CE-OST framework:
<img align="center" src="/visualization/framework.png">


### 3.1 Contour Emphasis

Make sure to create these folders ```contrast, contrast_temp, white, white_temp``` by the ```mkdir``` command or adjust in the source below.

```
CUDA_VISIBLE_DEVICES=0 python contour_emphasis/add_bd_grid.py \
                    --dataset_path ./path/to/dataset/ \
                    --output_path ./path/to/output/dir/
```

### 3.2 Camouflage Instance Segmentation
Initial parameters:
```
OUTPUT_DIR=./path/to/output/dir/
config=configs/<CONFIG_FILE_NAME>.yaml
WEIGHT=weights/<WEIGHT_FILE_NAME>.pth

cfg_MODEL='
SOLVER.IMS_PER_BATCH 1
DATALOADER.NUM_WORKERS 0
'
```

### Training

```
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
                    --num-gpus 1 \
                    --dist-url auto \
                    --num-machines 1 \
                    --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
```

### Testing

```
CUDA_VISIBLE_DEVICES=0 python tools/train_net.py \
                    --eval-only \
                    --num-gpus 1 \
                    --dist-url auto \
                    --num-machines 1 \
                    --opts MODEL.WEIGHTS ${WEIGHT} OUTPUT_DIR ${OUTPUT_DIR} ${cfg_MODEL}
```

The whole script commands can be found in `./scripts/run.sh`.

**Released checkpoints and results:**

We provide the checkpoints of our final model on PVT backbone under two versions of Grid-Addition mechanism:

| Model      | CE-OST Color Contrast | CE-OST Brightness Addition |
| ------------- |:---------------------:|:--------------------------:|
| PVTv2-B2-Li-COD10K   |   [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/Ef3G4jOL7edHvJO-5QYSo14BaiCeF4w6Qxq5uSc0luXIlA?e=uJbL3y)    |     [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/EZKT6lvlQFBLrEWF_2X39SEB1UAmMaTK0mWLxt4MJvOTtg?e=YUazK2)       |
| PVTv2-B2-Li-CAMO++   |   [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/EYLEJmYPsahCjqxIE6AvN5cB_2XrzGWKKG5sbgxxlYXr2w?e=a53oKR)    |     [link](https://uithcm-my.sharepoint.com/:u:/g/personal/danhnt_16_ms_uit_edu_vn/ETIBu8EsdJdPpCT0vn9B-LoBcY0Eep-HHcFbE9d6kOOB2A?e=kJgg4S)       |

## 4. Visualization

<p align="center">
  <img width="800" src="/visualization/visualization.png">
</p>

## Citation
Please use the following bibtex to cite this repository:
```
@inproceedings{nguyen2023ost,
  title={CE-OST: Contour Emphasis for One-Stage Transformer-based Camouflage Instance Segmentation},
  author={Nguyen, Thanh-Danh and Luu, Duc-Tuan and Nguyen, Vinh-Tiep and Ngo, Thanh Duc},
  booktitle={2023 International Conference on Multimedia Analysis and Pattern Recognition (MAPR)},
  pages={1--6},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgements

[OSFormer](https://github.com/PJLallen/OSFormer.git) [HED](https://github.com/s9xie/hed.git) [Detectron2](https://github.com/facebookresearch/detectron2.git) 