
################## import
import glob
import numpy as np
#read imagesis with: PIL, cv2, keras
from PIL import Image
from keras.preprocessing import image
from matplotlib import pyplot as plt
import pandas as pd

from keras.applications.imagenet_utils import preprocess_input

from keras.applications import VGG16,ResNet50
from collections import Counter
import xgboost as xgb

from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

import bcolz
import pickle
################### functions
def do_clip(arr, mx):
    clipped = np.clip(arr, (1-mx)/1, mx)
    return clipped/clipped.sum(axis=1)[:, np.newaxis]



def save_array(fname, arr):
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()


def load_array(fname):
    return bcolz.open(fname)[:]


def save_pickle(fname, obj):
    with open(fname, "wb") as output_file:
        pickle.dump(obj, output_file)

def load_pickle(fname):
    with open(fname, "rb") as input_file:
        return pickle.load(input_file)

################### reading/loading

def load_train_data_from_disk():
	#image: full/227/224
	#data: full/thin
	pass
	return imgs,y

################### preprocessing

################### models

def get_model():
	#type: vgg,resnet,squeezenet
	pass

################### train

################### predict

################### submit