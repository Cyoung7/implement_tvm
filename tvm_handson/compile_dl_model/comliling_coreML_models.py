import nnvm
import tvm
import coremltools as cm
import numpy as np
from PIL import Image


def download(url, path, overwrite=False):
    import os
    if os.path.isfile(path) and not overwrite:
        print('File {} existed, skip.'.format(path))
        return
    print('Downloading from url {} to {}'.format(url, path))
    import urllib.request
    urllib.request.urlretrieve(url, path)


model_url = 'https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel'
model_file = 'mobilenet.mlmodel'
download(model_url, model_file)
# now you mobilenet.mlmodel on disk
mlmodel = cm.models.MLModel(model_file)

sym,params = nnvm.frontend.from_coreml(mlmodel)

# load a test image
from PIL import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
download(img_url, 'cat.png')
img = Image.open('cat.png').resize((224, 224))
image = np.asarray(img)
image = image.transpose((2,0,1))
x = image[np.newaxis,:]


import nnvm.compiler
target = 'cuda'
shape_dict = {'image':x.shape}
graph,lib,params = nnvm.compiler.build(sym,target,shape_dict,params=params)

#execute on tvm
from tvm.contrib import graph_runtime
ctx = tvm.gpu(0)
dtype = 'float32'
m = graph_runtime.create(graph,lib,ctx)
m.set_input('image',tvm.nd.array(x.astype(dtype)))
m.set_input(**params)
m.run()
out_shape = (1000,)
tvm_output = m.get_output(0,tvm.nd.empty(out_shape,dtype)).asnumpy()
top1 = np.argmax(tvm_output)

synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                      '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                      '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                      'imagenet1000_clsid_to_human.txt'])
synset_name = 'synset.txt'
download(synset_url, synset_name)
with open(synset_name) as f:
    synset = eval(f.read())
print('Top-1 id', top1, 'class name', synset[top1])

print('hello')