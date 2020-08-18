# -*- coding: UTF-8 -*-
import argparse
import os
import tensorflow as tf
from tensorpack import *
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.tfutils.symbolic_functions import prediction_incorrect
from tensorpack.tfutils.summary import add_moving_summary
import cv2
import numpy as np
import squeezenet as net
from tensorflow.contrib.slim.python.slim.nets import inception_v3
slim = tf.contrib.slim

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_float('learning_rate', 0.1, '')
flags.DEFINE_string('path', '', '')
flags.DEFINE_string('path_label', '', '')
flags.DEFINE_string('finetune_path', '', '')
flags.DEFINE_integer('stop_epoch', 50, '')
flags.DEFINE_string('log_dir', '', '')
flags.DEFINE_string('path_test', '', '')
flags.DEFINE_string('path_label_test', '', '')
tf.logging.set_verbosity(tf.logging.INFO)

os.environ['TENSORPACK_TRAIN_API'] = 'v2'  # DO NOT change it
IMAGE_SIZE = 227  # the resized image w and h
IMAGE_SIZE_RESIZE = 249  # the resized image w and h
# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.
CLASSES = 8  # the softmax classes
GPU_NUM = 4  # the default gpu num


# Model class and YelpDateFlow class are the tensorpack's model and dataflow define,you can get the info from
# http://tensorpack.readthedocs.io/en/latest/tutorial/index.html#user-tutorials


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 3], 'input'),
                InputDesc(tf.int32, [None,], 'label')]

    def _build_graph(self, inputs):
        x, y_ = inputs
        y = net.squeezenet(x, True, 0.999, CLASSES)
        #with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        #  y, endpoints = inception_v3.inception_v3(x, CLASSES, True)
        #y_1 = tf.one_hot(tf.to_int32(y_), CLASSES)
        #class_weights = tf.constant([[1.0,1.0,1.0,1.0,1.0,1.0,2.0,3.0]])
        #weights = tf.gather(class_weights, y_)
        #y = tf.multiply(y,weights)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=y)
        #cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y, weights=weights)
        self.cost = tf.reduce_mean(cross_entropy, name='cross_entropy_loss')
        #y_2 = tf.reshape(y_, [-1,])
        acc = tf.to_float(tf.nn.in_top_k(y, y_, 1))
        acc = tf.reduce_mean(acc, name='accuracy')
        summary.add_moving_summary(acc)
        #correct_prediction = tf.equal(tf.cast(tf.argmax(y, 1), tf.int32), y_)
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuary_top1')
        #wrong = prediction_incorrect(y, y_, 1, name='wrong-top1')
        #add_moving_summary(tf.reduce_mean(wrong, name='train-wrong-top1'))

    def _get_optimizer(self):
       #lr = tf.get_variable('learning_rate', initializer=0.0005, trainable=False)
        # Create an optimizer that performs gradient descent.
        #return tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY,
        #                            momentum=RMSPROP_MOMENTUM,
        #                            epsilon=RMSPROP_EPSILON)
        lr = tf.get_variable('learning_rate', initializer=0.025, trainable=False)
        return tf.train.AdamOptimizer(lr, epsilon=1e-3)



class YelpDateFlow(RNGDataFlow):
    def __init__(self, tfrecord_path):
        self.tfrecord_path = tfrecord_path
        #for line in open(lable_path):
            #temp_str = line.split('\t', 1)
            #self.file_list.append((os.path.join(dir_path, temp_str[0]), int(temp_str[1])))

    #def size(self):
        #return len(self.file_list)

    #def get_data(self):
        #idxs = np.arange(len(self.file_list))
        #self.rng.shuffle(idxs)
        #for k in idxs:
            #fname, label = self.file_list[k]
            #im = cv2.imread(fname, cv2.IMREAD_COLOR)
           #label = np.reshape(label, [1,])
            #assert im is not None, fname
            #yield [im, label]

    def get_data(self):
        queue = tf.train.string_input_producer([self.tfrecord_path])
        reader = tf.TFRecordReader()
        _,serialized_example = reader.read(queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
          'pic_data': tf.FixedLenFeature([], tf.string),  
          'picid': tf.FixedLenFeature([], tf.int64), 
          'shopid':tf.FixedLenFeature([], tf.int64),  
          'pictype': tf.FixedLenFeature([], tf.int64),
          'shoptype': tf.FixedLenFeature([], tf.int64) 
                    }
            )
        image = tf.decode_raw(features['pic_data'],tf.uint8)
        #height = tf.cast(features['height'], tf.int64)
        #width = tf.cast(features['width'], tf.int64)
        #image = tf.reshape(image,[32,32,3])
        #image = tf.cast(image, tf.float32)
        #image = tf.image.per_image_standardization(image)
        label = tf.cast(features['pictype'], tf.int64)
        #print(image,label)
        return image,label


def get_data():
    augmentors = [
        #imgaug.Resize(IMAGE_SIZE_RESIZE),
        #imgaug.Rotation(max_deg=10),
        #imgaug.RandomApplyAug(imgaug.GaussianBlur(3), 0.5),
        #imgaug.Brightness(30, True),
        #imgaug.Gamma(),
        #imgaug.Contrast((0.8, 1.2), True),
        #imgaug.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        #imgaug.RandomApplyAug(imgaug.JpegNoise(), 0.8),
        #imgaug.RandomApplyAug(imgaug.GaussianDeform(
        #    [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
        #     (IMAGE_SIZE, IMAGE_SIZE), 0.2, 3), 0.1),
        #imgaug.Flip(horiz=True),
        imgaug.Resize(IMAGE_SIZE),
        #imgaug.MapImage(lambda x: x - 128),
    ]
    ds = YelpDateFlow(FLAGS.path)
    ds = AugmentImageComponent(ds, augmentors)
    ds = PrefetchDataZMQ(ds, nr_proc=4)
    ds = BatchData(ds, FLAGS.batch_size)
    return ds

def get_validation_data():
    augmentors = [
        imgaug.Resize(IMAGE_SIZE),
        #imgaug.MapImage(lambda x: x - 128),
    ]

    ds = YelpDateFlow(FLAGS.path_test)
    ds = AugmentImageComponent(ds, augmentors)
    ds = PrefetchDataZMQ(ds, nr_proc=4)
    ds = BatchData(ds, FLAGS.batch_size)
    return ds

def main(_):
    ### set the logdir
    ### an action of ("k","b","d","n","q") to be performed as k (keep) / b (backup) / d (delete) / n (new) / q (quit)
    logger.set_logger_dir(FLAGS.log_dir, 'k')
    dataset_train = get_data()
    dataset_val =  get_validation_data()
    config = TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        max_epoch=FLAGS.stop_epoch,
        session_init=SaverRestore(FLAGS.finetune_path),
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_val, ScalarStats(['cross_entropy_loss', 'accuracy'])),
            ScheduledHyperParamSetter('learning_rate',
                                      [(10, 0.01), (35, 0.001), (45, 0.0001), (55,0.00001), (65,0.000001)]),
            HumanHyperParamSetter('learning_rate'),
        ],
        monitors=[
            TFEventWriter(),
            ScalarPrinter()
        ]
    )

    gpu_num_by_env = os.getenv('CUDA_VISIBLE_DEVICES')
    print(gpu_num_by_env)
    if len(gpu_num_by_env) > 0:
        gpu_size = len(gpu_num_by_env.split(','))
    else:
        gpu_size = GPU_NUM
    print(gpu_size)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(gpu_size))


if __name__ == "__main__":
    tf.app.run()
