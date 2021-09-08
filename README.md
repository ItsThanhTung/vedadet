# Auto Label Using TinaFace


------------
## Docker.
docker pull itsthanhtung/tina-face:2
## Data file
* [data](./data)
   * [data/train2017](./data/train2017)
   * [data/out_annotation](./data/out_annotation)
   
Mount data directory in local to container directory: /workspace/vedadet/data

Example: 
```shell
docker run -it \
		--gpus all \
		--name tina \
		--mount type=bind,source="home/tung/data"/target,target=/workspace/vedadet/data \
  		itsthanhtung/tina-face:2
```	

### Run
```shell
CUDA_VISIBLE_DEVICES="0" python tools/infer.py --config ./configs/infer/tinaface/tinaface_r50_fpn_gn_dcn.py --path data/train2017 --out_path data/out_annotation
```

--out_img_path(optional) for result visualization

