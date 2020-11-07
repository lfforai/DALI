from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
import os.path

# test_data_root = os.environ['DALI_EXTRA_PATH']
test_data_root="/lf_tool/data/DALI_extra/" # you need to change it to your self env_path where you load DALI_EXTRA data
db_folder = os.path.join(test_data_root, 'db', 'lmdb')

class CaffePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(CaffePipeline, self).__init__(batch_size,
                                            num_threads,
                                            device_id)
        self.input = ops.CaffeReader(path = db_folder)
        self.decode= ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            dtype = types.FLOAT,
                                            crop = (224, 224),
                                            mean = [0., 0.4, 0.],
                                            std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.iter = 0

    # thera is a very import function ops.CropMirrorNormalize:
    # crop defined (crop_H,crop_W) ,real postion is crop_pos_x_real=crop_x_norm*(W-crop_W)
    # it means :      1--------2---------------------------------3
    #                     *(0.5)
    #                  *(0.1)
    #                        *(0.9)
    #                 1-2:W-crop_W
    #                 2-3:crop_w
    #                 1-3:image wight=W
    #                 *:crop_pos_x=crop_x_norm*(W-crop_W): crop_x_norm=ops.Uniform

    def define_graph(self):
        self.jpegs, self.labels = self.input()
        images = self.decode(self.jpegs)
        output = self.cmnp(images, crop_pos_x = self.uniform(),
                           crop_pos_y = self.uniform())
        return (output, self.labels)

    def iter_setup(self):
        pass

batch_size = 16

pipe = CaffePipeline(batch_size=batch_size, num_threads=4, device_id = 0)
pipe.build()
pipe_out = pipe.run()

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# %matplotlib inline

def show_images(image_batch):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        img_chw = image_batch.at(j)
        img_hwc = np.transpose(img_chw, (1,2,0))/255.0
        plt.imshow(img_hwc)

images, labels = pipe_out
show_images(images.as_cpu())
plt.show()

#if you want to know np.tanspose(),use blow exmaple
#CHW is transformed to HWC,plt.imshow()  can  only show picture in the type of HWC
import tensorflow as tf
x=np.arange(12).reshape((2,2,3))
print("HWC：",x)
x=tf.constant(x)
x=tf.transpose(x,(2,0,1))
print("CHW:",x)
