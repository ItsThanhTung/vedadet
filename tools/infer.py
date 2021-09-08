import argparse

import cv2
import numpy as np
import torch
import os
from vedacore.image import imread, imwrite
from vedacore.misc import Config, color_val, load_weights
from vedacore.parallel import collate, scatter
from vedadet.datasets.pipelines import Compose
from vedadet.engines import build_engine


def parse_args():
    parser = argparse.ArgumentParser(description='Infer a detector')
    parser.add_argument('--config', help='config file path')
    parser.add_argument('--path', help='image file name')
    parser.add_argument('--out_path', help='image out path')
    parser.add_argument('--out_img_path', help='image out path', default= None)

    args = parser.parse_args()
    return args


def prepare(cfg):
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    engine = build_engine(cfg.infer_engine)

    engine.model.to(device)
    load_weights(engine.model, cfg.weights.filepath)

    data_pipeline = Compose(cfg.data_pipeline)
    return engine, data_pipeline, device

def box_corner_to_center(img, left_top, right_bottom):
    x1, y1, x2, y2 = left_top[0], left_top[1], right_bottom[0], right_bottom[1]
    height, width = img.shape[0], img.shape[1]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = str(80) + " " + str(cx/width) + " " + str(cy/height) + " " + str(w/width) + " " + str(h/height) + "\n"
    return boxes

def plot_result(result, imgfp, class_names, outfp='out.jpg', img_path = None):
    font_scale = 0.5
    bbox_color = 'green'
    text_color = 'green'
    thickness = 1

    bbox_color = color_val(bbox_color)
    text_color = color_val(text_color)
    img = imread(imgfp)

    bboxes = np.vstack(result)
    labels = [
        np.full(bbox.shape[0], idx, dtype=np.int32)
        for idx, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    txt_out_file = outfp[:-4] + ".txt"

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox[:4].astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        result_line = box_corner_to_center(img, left_top, right_bottom)
        file = open(txt_out_file, "a") 
        file.write(result_line) 
        file.close() 
        if img_path != None:
          cv2.rectangle(img, left_top, right_bottom, bbox_color, thickness)
          label_text = class_names[
              label] if class_names is not None else f'cls {label}'
          if len(bbox) > 4:
              label_text += f'|{bbox[-1]:.02f}'
          cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                      cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color)
    if img_path != None:
      imwrite(img, img_path)


def main():

    args = parse_args()
    cfg = Config.fromfile(args.config)
    img_path = args.path
    class_names = cfg.class_names
    out_path = args.out_path
    out_img_path = args.out_img_path
    engine, data_pipeline, device = prepare(cfg)
    # count = 0 
    for file in os.listdir(img_path):
      # count = count + 1
      # print(count)
      imgname = os.path.join(img_path, file)
      out_file = os.path.join(out_path, file)
      out_image = None
      if out_img_path != None:
        out_image = os.path.join(out_img_path, file)
      

      data = dict(img_info=dict(filename=imgname), img_prefix=None)

      data = data_pipeline(data)
      data = collate([data], samples_per_gpu=1)
      if device != 'cpu':
          # scatter to specified GPU
          data = scatter(data, [device])[0]
      else:
          # just get the actual data from DataContainer
          data['img_metas'] = data['img_metas'][0].data
          data['img'] = data['img'][0].data
      result = engine.infer(data['img'], data['img_metas'])[0]
      plot_result(result, imgname, class_names, outfp= out_file, img_path = out_image )


if __name__ == '__main__':
    main()
