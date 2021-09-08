## Installation
### Requirements

- Linux
- Python 3.7+
- PyTorch 1.6.0 or higher
- CUDA 10.2 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.2
- PyTorch 1.6.0
- Python 3.8.5

### Install vedadet

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedadet python=3.8.5 -y
conda activate vedadet
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the vedadet repository.

```shell
git clone https://github.com/ItsThanhTung/vedadet.git
cd vedadet
vedadet_root=${PWD}
```

d. Install vedadet.

```shell
pip install -r requirements/build.txt
pip install -v -e .
```


## Inference
a. Download model

Download Pretrained model: https://drive.google.com/uc?id=1zU738coEVDBkLBUa4hvJUucL7dcSBT7v (tinaface_r50_fpn_gn_dcn)

b. Config

Modify model path in config file configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py

b. Inference

```shell
CUDA_VISIBLE_DEVICES="0" python tools/infer.py --config configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py --path train2017 --out_path out_annotation --out_img_path out_image
```

