import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import glob

from Mask.config import Config
import Mask.utils as utils
import Mask.model as modellib
import Mask.visualize as visualize
import tensorflow as tf

np.set_printoptions(threshold=np.inf)
tf.logging.set_verbosity(tf.logging.ERROR)
# path of the trained model
dir_path = os.path.dirname(os.path.realpath(__file__))
MODEL_DIR = dir_path + "/models/"
MODEL_PATH = "models/MAKRCNN/mask_rcnn_moles_0033.h5"


class CocoConfig(Config):
    ''' 
    MolesConfig:
        Contain the configuration for the dataset + those in Config
    '''
    NAME = "moles"
    NUM_CLASSES = 1 + 2 # background + (malignant , benign)
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 3

# create and instance of config
config = CocoConfig()
class_names = ["BG", "malign", "benign"]
# take the trained model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)
img = open("Data/Images/1_malign.jpg", 'rb').read()
nparr = np.fromstring(img, np.uint8)
img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
# background + (malignant , benign)

height, width = img.shape[0], img.shape[1]
r = model.detect([img])[0]

image_mask = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                            class_names, r['scores'])
image_mask = cv2.resize(image_mask, (height, width))
if len(r['class_ids']) > 0:
    class_name = class_names[r['class_ids'][0]]
else:
    class_name = "unsure"
print(class_name)

