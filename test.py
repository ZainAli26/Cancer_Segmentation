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
#MODEL_PATH = input("Insert the path of your trained model [ Like models/moles.../mask_rcnn_moles_0030.h5 ]: ")
#if os.path.isfile(MODEL_PATH) == False:
#    raise Exception(MODEL_PATH + " Does not exists")
MODEL_PATH = "models/moles20190910T1743/mask_rcnn_moles_0033.h5"
# path of Data that contain Descriptions and Images
#path_data = input("Insert the path of Data [ Link /home/../ISIC-Archive-Downloader/Data/ ] : ")
path_data = "Data/Images"
#if not os.path.exists(path_data):
#    raise Exception(path_data + " Does not exists")


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

# take the trained model
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_weights(MODEL_PATH, by_name=True)

# background + (malignant , benign)
class_names = ["BG", "malign", "benign"]

# find the largest number of image that you download
segmentation_img_path = "/home/zain/Piyarse/Data/Segmentation"
orig_img_path = "Data/Images"
out_path = "Data/OUT/"
#all_desc_path = glob.glob(path_data + "Descriptions/ISIC_*")
file_ = open('out.txt','w')
i = 0
tot = 0
for filename in os.listdir(orig_img_path):
    class_name = filename.split('.')[0].split('_')[1]
    img = cv2.imread(os.path.join(orig_img_path, filename))
    img = cv2.resize(img, (128, 128))

    if img is None:
        continue

    # ground truth of the class
    i += 1
    # predict the mask, bounding box and class of the image
    r = model.detect([img])[0]
    #file_.write(class_name + ',' + r['rois'] + ',' + r['masks'] + ',' + r['class_ids']+ ',' +class_names+ ',' + r['scores'])
    #file_.write('\n')
    image_mask = visualize.display_instances(img, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    cv2.imwrite(out_path + filename,image_mask)
    print(filename, class_name, r['class_ids'], r['scores'])
    if len(r['class_ids']) > 0:
        if class_name==class_names[r['class_ids'][0]]:
            tot+= 1
print("Accuracy: ", tot/i)

