from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
from time import time
import os.path
help(ops)
# test_data_root = os.environ['DALI_EXTRA_PATH']
test_data_root="/lf_tool/data/DALI_extra/"
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
annotations_file = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

num_gpus = 1
batch_size = 16

class COCOPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(COCOPipeline, self).__init__(batch_size, num_threads, device_id, seed = 15)
        self.input = ops.COCOReader(file_root = file_root, annotations_file = annotations_file,
                                    shard_id = device_id, num_shards = num_gpus, ratio=True)
        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)

    def define_graph(self):
        inputs, bboxes, labels = self.input()
        images = self.decode(inputs)
        return (images, bboxes, labels)

start = time()
pipes = [COCOPipeline(batch_size=batch_size, num_threads=2, device_id = device_id)  for device_id in range(num_gpus)]
for pipe in pipes:
    pipe.build()
total_time = time() - start
print("Computation graph built and dataset loaded in %f seconds." % total_time)

pipe_out = [pipe.run() for pipe in pipes]

images_cpu = pipe_out[0][0].as_cpu()
bboxes_cpu = pipe_out[0][1]
labels_cpu = pipe_out[0][2]

bboxes = bboxes_cpu.at(4)
bboxes
print(bboxes)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

img_index = 4

img = images_cpu.at(img_index)

H = img.shape[0]
W = img.shape[1]

fig,ax = plt.subplots(1)

ax.imshow(img)
bboxes = bboxes_cpu.at(img_index)
labels = labels_cpu.at(img_index)
categories_set = set()
for label in labels:
    categories_set.add(label[0])

category_id_to_color = dict([ (cat_id , [random.uniform(0, 1) ,random.uniform(0, 1), random.uniform(0, 1)]) for cat_id in categories_set])

for bbox, label in zip(bboxes, labels):
    rect = patches.Rectangle((bbox[0]*W,bbox[1]*H),bbox[2]*W,bbox[3]*H,linewidth=1,edgecolor=category_id_to_color[label[0]],facecolor='none')
    ax.add_patch(rect)

plt.show()

