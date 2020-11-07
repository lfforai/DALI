from subprocess import call
import os.path

# test_data_root = os.environ['DALI_EXTRA_PATH']
test_data_root="/lf_tool/data/DALI_extra/" # you need to change it to your self env_path where you load DALI_EXTRA data
tfrecord = os.path.join(test_data_root, 'db', 'tfrecord', 'train')
tfrecord_idx = "idx_files/train.idx"
tfrecord2idx_script = "tfrecord2idx"

if not os.path.exists("idx_files"):
    os.mkdir("idx_files")

if not os.path.isfile(tfrecord_idx):
    call([tfrecord2idx_script, tfrecord, tfrecord_idx])

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import numpy as np
import matplotlib.pyplot as plt

class TFRecordPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(TFRecordPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id)
        self.input = ops.TFRecordReader(path = tfrecord,
                                        index_path = tfrecord_idx,
                                        features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                                    'image/class/label':         tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                                    'image/class/text':          tfrec.FixedLenFeature([ ], tfrec.string, ''),
                                                    'image/object/bbox/xmin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                                    'image/object/bbox/ymin':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                                    'image/object/bbox/xmax':    tfrec.VarLenFeature(tfrec.float32, 0.0),
                                                    'image/object/bbox/ymax':    tfrec.VarLenFeature(tfrec.float32, 0.0)})
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu", resize_shorter = 256.)
        self.cmnp = ops.CropMirrorNormalize(device = "gpu",
                                            dtype = types.FLOAT,
                                            crop = (224, 224),
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.iter = 0

    def define_graph(self):
        inputs = self.input()
        images = self.decode(inputs["image/encoded"])
        resized_images = self.resize(images)
        output = self.cmnp(resized_images, crop_pos_x = self.uniform(),
                           crop_pos_y = self.uniform())
        return (output, inputs["image/class/text"])

    def iter_setup(self):
        pass

batch_size = 16

pipe = TFRecordPipeline(batch_size=batch_size, num_threads=4, device_id = 1)
pipe.build()
pipe_out = pipe.run()

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
# %matplotlib inline

def show_images(image_batch, labels):
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        ascii = labels.at(j)
        plt.title("".join([chr(item) for item in ascii]))
        img_chw = image_batch.at(j)
        img_hwc = np.transpose(img_chw, (1,2,0))/255.0
        plt.imshow(img_hwc)

images, labels = pipe_out
show_images(images.as_cpu(), labels)
plt.show()

#if you want to know np.tanspose(),use blow exmaple
#CHW is transformed to HWC,plt.imshow()  can  only show picture in the type of HWC
import tensorflow as tf
x=np.arange(12).reshape((2,2,3))
print("HWC：",x)
x=tf.constant(x)
x=tf.transpose(x,(2,0,1))
print("CHW:",x)