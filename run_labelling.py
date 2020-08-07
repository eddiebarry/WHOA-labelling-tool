#!/usr/bin/python

import sys
import os
from modelbuilder import ModelFinder

name_project   = sys.argv[1]
model_dir      = "models/"+name_project+"/model.py"
train_data_dir = "data/train/"+name_project+"/"
test_data_dir  = "data/test/"+name_project+"/"
weight_dir     = "/kaggle/input/multilabel-text-class/model.h5"
batch_size = 16

# read data
train_dirs = []
for data in os.listdir(train_data_dir):
    fp = os.path.join(train_data_dir,data)
    train_dirs.append(fp)

test_dirs = []
for data in os.listdir(test_data_dir):
    fp = os.path.join(test_data_dir,data)
    test_dirs.append(fp)

print(test_dirs)
print('*'*80)
print(train_dirs)
print("data loaded from directories")

# give data to model
model = ModelFinder(name_project).getModel()
model.set_data(train=train_dirs, test=test_dirs, batch_size=batch_size, \
    weight_dir=weight_dir)
print("model loaded")

# train data
# model.train(lr=1e-3,optim="Adam",loss="BCE")
print("model trained")

# label data in test dir
save_pth = os.path.join("./data/results", name_project)
model.test(save_dir=save_pth)
print("predictions saved")