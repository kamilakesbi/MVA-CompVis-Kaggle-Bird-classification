# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from glob import glob
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# Functions used to crop objects: 

def crop_object(image, box, visualize = True):
  """Crops a box from an image
  Inputs:
    image: numpy nd.array
    box: one box from Detectron2 pred_boxes
  Output : 
    crop_image : numpy nd.array containg he croped image
  """

  x_top_left,y_top_left, x_bottom_right, y_bottom_right  = box
  y_max, x_max, _ = image.shape

  if (int(y_top_left) - int(y_bottom_right)) > (int(x_bottom_right) - int(x_top_left)): 
    size = int(y_top_left) - int(y_bottom_right)
    y_1 = max(0,-5 + int(y_top_left))
    y_2 = min(int(y_bottom_right) + 5, y_max -1 )
    x_1 = max(0,- 5 + int(x_top_left))
    x_2 = min(int(5 + x_top_left + size), x_max  -1 )
    crop_image = image[ y_1 : y_2 , x_1 : x_2 ]
  else : 
    size = int(x_bottom_right) - int(x_top_left -1)
    y_1 = max(0, -5 + int(y_top_left))
    y_2 = min(int(5 + y_top_left + size), y_max -1  )
    x_1 = max(0, - 5 + int(x_top_left))
    x_2 = min(5 + int(x_bottom_right), x_max -1  )
    crop_image = image[ y_1: y_2 , x_1 :x_2]

  if visualize == True : 
    cv2_imshow(crop_image)
  return crop_image


def bird_extraction(predictor, image_path) :
  """ Extract the most confident bird box (if exists) from input image
  Inputs:
    image_path : str, path to image
    predictor : detectron2 predictor
  Output : 
    im : numpy nd.array containg the bird box or the input image
  """

  im = cv2.imread(image_path)
  outputs = predictor(im)

  boxes_classes = outputs["instances"].pred_classes
  bird_boxes_indexes = [i for i, j in enumerate(boxes_classes.tolist()) if j == 14]
  if len(bird_boxes_indexes) !=0 : 
    bird_scores = [outputs["instances"].scores[i] for i in bird_boxes_indexes]
    highest_bird_score_index = bird_scores.index(max(bird_scores))
    bird_boxe = outputs["instances"].pred_boxes[highest_bird_score_index]
    bird_boxe = list(list(bird_boxe)[0].detach().cpu().numpy())
    return crop_object(im, bird_boxe, visualize = True)
  else : 
    print('error', image_path)
    return im
