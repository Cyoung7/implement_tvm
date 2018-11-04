# -*-coding:utf-8-*-
import nnvm
import tvm

# os and numpy
import numpy as np
import os.path

# Tensorflow imports
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util

# Tensorflow utility functions
import nnvm.testing.tf

# Base location for model related files.
repo_base = 'https://github.com/dmlc/web-data/raw/master/tensorflow/models/InceptionV1/'

# Test image
img_name = 'elephant-299.jpg'
image_url = os.path.join(repo_base, img_name)
model_name = 'classify_image_graph_def-with_shapes.pb'
model_url = os.path.join(repo_base, model_name)

# Image label map
map_proto = 'imagenet_2012_challenge_label_map_proto.pbtxt'
map_proto_url = os.path.join(repo_base, map_proto)

# Human readable text for labels
lable_map = 'imagenet_synset_to_human_label_map.txt'
lable_map_url = os.path.join(repo_base, lable_map)

# Target settings
# Use these commented settings to build for cuda.
#target = 'cuda'
#target_host = 'llvm'
#layout = "NCHW"
#ctx = tvm.gpu(0)
target = 'llvm'
target_host = 'llvm'
layout = None
ctx = tvm.cpu(0)

from mxnet.gluon.utils import download
store_path = './tf_model/'
download(image_url, store_path+img_name)
download(model_url, store_path+model_name)
download(map_proto_url, store_path+map_proto)
download(lable_map_url, store_path+lable_map)

with tf.gfile.FastGFile(store_path+model_name, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    graph = tf.import_graph_def(graph_def, name='')
    # Call the utility to import the graph definition into default graph.
    graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    with tf.Session() as sess:
        graph_def = nnvm.testing.tf.AddShapesToGraphDef(sess, 'softmax')


from PIL import Image
image = Image.open(store_path+img_name).resize((299, 299))
x = np.array(image)

sym, params = nnvm.frontend.from_tensorflow(graph_def, layout=layout)
print ("Tensorflow protobuf imported as nnvm graph")

import nnvm.compiler
shape_dict = {'DecodeJpeg/contents': x.shape}
dtype_dict = {'DecodeJpeg/contents': 'uint8'}
graph, lib, params = nnvm.compiler.build(sym, shape=shape_dict, target=target, target_host=target_host, dtype=dtype_dict, params=params)
print('build completed-------------------------------------------------------')
from tvm.contrib import graph_runtime
dtype = 'uint8'
m = graph_runtime.create(graph, lib, ctx)
print('create module competed-----------------------')
# set inputs
m.set_input('DecodeJpeg/contents', tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
# execute
m.run()
# get outputs
tvm_output = m.get_output(0, tvm.nd.empty(((1, 1008)), 'float32'))

predictions = tvm_output.asnumpy()
predictions = np.squeeze(predictions)

# Creates node ID --> English string lookup.
node_lookup = nnvm.testing.tf.NodeLookup(label_lookup_path=os.path.join(store_path, map_proto),
                                         uid_lookup_path=os.path.join(store_path, lable_map))

# Print top 5 predictions from TVM output.
top_k = predictions.argsort()[-5:][::-1]
for node_id in top_k:
    human_string = node_lookup.id_to_string(node_id)
    score = predictions[node_id]
    print('%s (score = %.5f)' % (human_string, score))

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(store_path+model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name='')
        # Call the utility to import the graph definition into default graph.
        graph_def = nnvm.testing.tf.ProcessGraphDefParam(graph_def)

def run_inference_on_image(image):
    """Runs inference on an image.

    Parameters
    ----------
    image: String
        Image file name.

    Returns
    -------
        Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    # Creates graph from saved GraphDef.
    create_graph()

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})

        predictions = np.squeeze(predictions)

        # Creates node ID --> English string lookup.
        node_lookup = nnvm.testing.tf.NodeLookup(label_lookup_path=os.path.join(store_path, map_proto),
                                                 uid_lookup_path=os.path.join(store_path, lable_map))

        # Print top 5 predictions from tensorflow.
        top_k = predictions.argsort()[-5:][::-1]
        print ("===== TENSORFLOW RESULTS =======")
        for node_id in top_k:
            human_string = node_lookup.id_to_string(node_id)
            score = predictions[node_id]
            print('%s (score = %.5f)' % (human_string, score))

run_inference_on_image (store_path+img_name)


