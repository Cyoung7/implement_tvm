import os
import zipfile
import tvm
import mxnet as mx
import cv2
import numpy as np

from nnvm import compiler
from nnvm.frontend import from_mxnet
from tvm.contrib.download import download
from tvm.contrib import graph_runtime
from mxnet.model import load_checkpoint

model_name = "ssd_resnet50_512"
model_file = "%s.zip" % model_name
test_image = "dog.jpg"
dshape = (1, 3, 512, 512)
dtype = "float32"
target = "llvm"
ctx = tvm.cpu()

model_url = "https://github.com/zhreshold/mxnet-ssd/releases/download/v0.6/" \
            "resnet50_ssd_512_voc0712_trainval.zip"
image_url = "https://cloud.githubusercontent.com/assets/3307514/20012567/" \
            "cbb60336-a27d-11e6-93ff-cbc3f09f5c9e.jpg"
inference_symbol_folder = "c1904e900848df4548ce5dfb18c719c7-a28c4856c827fe766aa3da0e35bad41d44f0fb26"
inference_symbol_url = "https://gist.github.com/kevinthesun/c1904e900848df4548ce5dfb18c719c7/" \
                       "archive/a28c4856c827fe766aa3da0e35bad41d44f0fb26.zip"

dir = "model"
if not os.path.exists(dir):
    os.makedirs(dir)
model_file_path = "%s/%s" % (dir, model_file)
test_image_path = "%s/%s" % (dir, test_image)
inference_symbol_path = "%s/inference_model.zip" % dir
download(model_url, model_file_path)
download(image_url, test_image_path)
download(inference_symbol_url, inference_symbol_path)

zip_ref = zipfile.ZipFile(model_file_path, 'r')
zip_ref.extractall(dir)
zip_ref.close()
zip_ref = zipfile.ZipFile(inference_symbol_path)
zip_ref.extractall(dir)
zip_ref.close()