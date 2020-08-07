#!/usr/bin/python

import sys
import os
from modelbuilder import ModelFinder

name_project   = sys.argv[1]
model_dir      = "models/"+name_project+"/model.py"
train_data_dir = "data/train/"+name_project+"/"
test_data_dir  = "data/test/"+name_project+"/"

# weight_dir     = "models/vac_safety/weights/model.hdf5"
weight_dir     = None
batch_size = 16

# read data
# train_dirs = []
# for data in os.listdir(train_data_dir):
#     fp = os.path.join(train_data_dir,data)
#     train_dirs.append(fp)

# test_dirs = []
# for data in os.listdir(test_data_dir):
#     fp = os.path.join(test_data_dir,data)
#     test_dirs.append(fp)

train_dirs = "/Users/edgarmonis/Desktop/WHO/labelling_tool/data/train/vac_safety/train_vac_xcl.csv"
test_dirs  = "/Users/edgarmonis/Desktop/WHO/labelling_tool/data/test/vac_safety/test_vac_xcl.csv"
print("data loaded from directories")

# give data to model
model = ModelFinder(name_project).getModel()
model.set_data(train=train_dirs, test=test_dirs, batch_size=batch_size, \
    weight_dir=weight_dir,data_name="vac_data", num_classes=26)
print("model loaded")

# train data
model.train(lr=1e-3,optim="Adam",loss="BCE")
print("model trained")

# label data in test dir
save_pth = os.path.join("./data/results", name_project)
model.test(save_dir=save_pth)
print("predictions saved")