import torch, detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from .utils import save_image, show_image
from constants.paths import DATA_DIR

input_image = str(DATA_DIR / 'samples' / 'input_2.png')
config = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
print(input_image)

def detect():
  im = cv2.imread(input_image)

  cfg = get_cfg()
  
  cfg.merge_from_file(model_zoo.get_config_file(config))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2  # set threshold for this model
  
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config)
  predictor = DefaultPredictor(cfg)
  outputs = predictor(im)

  # look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
  print(outputs["instances"].pred_classes)
  print(outputs["instances"].pred_boxes)

  v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  save_image(out.get_image()[:, :, ::-1])