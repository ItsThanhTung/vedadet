# Auto Label Using TinaFace


------------
## Docker.
docker pull itsthanhtung/tina-face:2
## Data file
* [data](./data)
   * [data/train2017](./data/train2017)
   * [data/out_annotation](./data/out_annotation)
   
Mount data file in local to container directory /workspace/vedadet/data

Example: 
```shell
docker run -it \
		--gpus all \
		--name tina \
		--mount type=bind,source="home/tung/data"/target,target=/workspace/vedadet/data \
  		itsthanhtung/tina-face:2
```	

wget https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5x6.pt
### Yolov5m6 retrained model on License Plate dataset
gdown https://drive.google.com/file/d/1wr4ObBBjMQ-PX1mClTNtn8fJRLSmLYZD/view?usp=sharing -O model.pt
## Detection (auto label)
python3 detect.py --weights model.pt --img 1280 --source image_paths



