from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os.path

# test_data_root = os.environ['DALI_EXTRA_PATH']
test_data_root="/lf_tool/data/DALI_extra/"
base = os.path.join(test_data_root, 'db', 'recordio')

idx_files = [base + "/train.idx"]
rec_files = [base + "/train.rec"]
idx_files

class RecordIOPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(RecordIOPipeline, self).__init__(batch_size,
                                               num_threads,
                                               device_id)
        self.input = ops.MXNetReader(path = rec_files, index_path = idx_files)

        #mixed  is used in cpu and gpu togather

        self.decode = ops.ImageDecoder(device = "mixed", output_type = types.RGB)
        self.cmn = ops.CropMirrorNormalize(device = "gpu",
                                           dtype = types.FLOAT,
                                           crop = (224, 224),
                                           mean = [0., 0., 0.],
                                           std = [1., 1., 1.])
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.iter = 0

    def define_graph(self):
        inputs, labels = self.input(name="Reader")
        images = self.decode(inputs)
        output = self.cmn(images, crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        return (output, labels)

    def iter_setup(self):
        pass

batch_size = 16

pipe = RecordIOPipeline(batch_size=batch_size, num_threads=2, device_id = 0)
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