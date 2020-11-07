from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os.path
help(ops)

# test_data_root = os.environ['DALI_EXTRA_PATH']
BATCH_SIZE = 32
DEVICES = 1
#test_data_root = os.environ['DALI_EXTRA_PATH']
test_data_root="/lf_tool/data/DALI_extra/"
file_root = os.path.join(test_data_root, 'db', 'coco', 'images')
annotations_file = os.path.join(test_data_root, 'db', 'coco', 'instances.json')

class COCOPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus):
        super(COCOPipeline, self).__init__(batch_size, num_threads, device_id, seed = 15)
        self.input = ops.COCOReader(file_root = file_root, annotations_file = annotations_file,
                                    shard_id = device_id, num_shards = num_gpus, ratio=False, save_img_ids=True)
        #the example ratio=False ,see all info of COCOReader ,see help(ops)
        self.decode = ops.ImageDecoder(device = "cpu", output_type = types.RGB)
        self.resize = ops.Resize(device = "cpu",
                                 interp_type = types.INTERP_LINEAR)
        self.cmn = ops.CropMirrorNormalize(device = "cpu",
                                           dtype = types.FLOAT,
                                           crop = (224, 224),
                                           mean = [128., 128., 128.],
                                           std = [1., 1., 1.])
        self.res_uniform = ops.Uniform(range = (256.,480.))
        self.uniform = ops.Uniform(range = (0.0, 1.0))
        self.cast = ops.Cast(device = "cpu",
                             dtype = types.INT32)

    def define_graph(self):
        inputs, bboxes, labels, im_ids = self.input()
        images = self.decode(inputs)
        images = self.resize(images, resize_shorter = self.res_uniform())
        output = self.cmn(images, crop_pos_x = self.uniform(),
                          crop_pos_y = self.uniform())
        output = self.cast(output)
        return (output, bboxes, labels, im_ids)

pipes = [COCOPipeline(batch_size=BATCH_SIZE, num_threads=2, device_id = device_id, num_gpus = DEVICES) for device_id in range(DEVICES)]
import tensorflow as tf
import nvidia.dali.plugin.tf as dali_tf
import time

daliop = dali_tf.DALIIterator()

images = []
bboxes = []
labels = []
image_ids = []
for d in range(DEVICES):
    with tf.device('/cpu'): # tt.sparse only can be used in cpu
        image, bbox, label, id = daliop(pipeline = pipes[d],
                                        shapes = [(BATCH_SIZE, 3, 224, 224), (), (), ()],
                                        dtypes = [tf.int32, tf.float32, tf.int32, tf.int32], sparse = [False, True, True])
        images.append(image)
        bboxes.append(bbox)
        labels.append(label)
        image_ids.append(id)

print(bboxes[0])
print(labels[0])