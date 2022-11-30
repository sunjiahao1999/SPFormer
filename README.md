# SPFormer

NEWS:🔥SPFormer is accepted at AAAI2023!🔥

[Superpoint Transformer for 3D Scene Instance Segmentation](https://arxiv.org/abs/2211.15766)

Jiahao Sun, Chunmei Qing, Junpeng Tan, Xiangmin Xu

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/superpoint-transformer-for-3d-scene-instance/3d-instance-segmentation-on-scannetv2)](https://paperswithcode.com/sota/3d-instance-segmentation-on-scannetv2?p=superpoint-transformer-for-3d-scene-instance)  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/superpoint-transformer-for-3d-scene-instance/3d-instance-segmentation-on-s3dis)](https://paperswithcode.com/sota/3d-instance-segmentation-on-s3dis?p=superpoint-transformer-for-3d-scene-instance)

<img src="docs\SPFormer.png" />

## Introduction

​	Most existing methods realize 3D instance segmentation by extending those models used for 3D object detection or 3D semantic segmentation. However, these non-straightforward methods suffer from two drawbacks: 1) Imprecise bounding boxes or unsatisfactory semantic predictions limit the performance of the overall 3D instance segmentation framework. 2) Existing method requires a time-consuming intermediate step of aggregation. To address these issues, this paper proposes a novel end-to-end 3D instance segmentation method based on Superpoint Transformer, named as SPFormer. It groups potential features from point clouds into superpoints, and directly predicts instances through query vectors without relying on the results of object detection or semantic segmentation. The key step in this framework is a novel query decoder with transformers that can capture the instance information through the superpoint cross-attention mechanism and generate the superpoint masks of the instances. Through bipartite matching based on superpoint masks, SPFormer can implement the network training without the intermediate aggregation step, which accelerates the network. Extensive experiments on ScanNetv2 and S3DIS benchmarks verify that our method is concise yet efficient. Notably, SPFormer exceeds compared state-of-the-art methods by 4.3% on ScanNetv2 hidden test set in terms of mAP and keeps fast inference speed (247ms per frame) simultaneously.

<img src="docs\snapshot.png" alt="snapshot" style="zoom:50%;" />

The snapshot from ScanNetv2 benchmark testing server on 11/07/2022. SPFormer ranks top on the AP50 leadboard.

## Installation

Requirements

- Python 3.x
- Pytorch 1.10
- CUDA 10.x or higher

The following installation suppose `python=3.8` `pytorch=1.10` and `cuda=11.4`.

- Create a conda virtual environment

  ```
  conda create -n spformer python=3.8
  conda activate spformer
  ```

- Clone the repository

  ```
  git clone https://github.com/sunjiahao1999/SPFormer.git
  ```

- Install the dependencies

  Install [Pytorch 1.10](https://pytorch.org/)

  ```
  pip install spconv-cu114
  conda install pytorch-scatter -c pyg
  pip install -r requirements.txt
  ```

  Install segmentator from this [repo](https://github.com/Karbo123/segmentator) (We wrap the segmentator in ScanNet).

- Setup, Install spformer and pointgroup_ops.

  ```
  sudo apt-get install libsparsehash-dev
  python setup.py develop
  cd spformer/lib/
  python setup.py develop
  ```

## Data Preparation

### ScanNet v2 dataset

Download the [ScanNet](http://www.scan-net.org/) v2 dataset.

Put the downloaded `scans` and `scans_test` folder as follows.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
```

Split and preprocess data

```
cd data/scannetv2
bash prepare_data.sh
```

The script data into train/val/test folder and preprocess the data. After running the script the scannet dataset structure should look like below.

```
SPFormer
├── data
│   ├── scannetv2
│   │   ├── scans
│   │   ├── scans_test
│   │   ├── train
│   │   ├── val
│   │   ├── test
│   │   ├── val_gt
```

## Pretrained Model

Download [SSTNet](https://drive.google.com/file/d/1vucwdbm6pHRGlUZAYFdK9JmnPVerjNuD/view?usp=sharing) pretrained model (We only use the Sparse 3D U-Net backbone for training).

Move the pretrained model to checkpoints.

```
mkdir checkpoints
mv ${Download_PATH}/sstnet_pretrain.pth checkpoints/
```

## Training

```
python tools/train.py configs/spf_scannet.yaml
```

## Inference

Download [SPFormer](https://drive.google.com/file/d/1BKuaLTU3TFgekYAssSVxPO0sHWj-LGlH/view?usp=sharing) pretrain model and move it to checkpoints. Its performance on ScanNet v2 validation set is 56.3/73.9/82.9 in terms of mAP/mAP50/mAP25.

```
python tools/test.py configs/spf_scannet.yaml checkpoints/spf_scannet_512.pth
```

## Visualization

Before visualization, you need to write the output results of inference.

```
python tools/test.py configs/spf_scannet.yaml ${CHECKPOINT} --out ${SAVE_PATH}
```

After inference, run visualization by execute the following command. 

```
python tools/visualization.py --prediction_path ${SAVE_PATH}
```

You can visualize by Open3D or visualize saved `.ply` files on MeshLab. Arguments explaination can be found in `tools/visualiztion.py`.

## Citation

If you find this work useful in your research, please cite:

```
@misc{2211.15766,
Author = {Jiahao Sun and Chunmei Qing and Junpeng Tan and Xiangmin Xu},
Title = {Superpoint Transformer for 3D Scene Instance Segmentation},
Year = {2022},
Eprint = {arXiv:2211.15766},
}
```

## Ancknowledgement

Sincerely thanks for [SoftGroup](https://github.com/thangvubk/SoftGroup) and [SSTNet](https://github.com/Gorilla-Lab-SCUT/SSTNet) repos. This repo is build upon them.

