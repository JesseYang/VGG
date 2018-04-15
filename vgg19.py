#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet.py

import argparse
import os

import tensorflow as tf

from tensorpack import logger, QueueInput
from tensorpack.models import *
from tensorpack.callbacks import *
from tensorpack.train import (
    TrainConfig, SyncMultiGPUTrainerReplicated, launch_train_with_config)
from tensorpack.dataflow import FakeData
from tensorpack.tfutils import argscope, get_model_loader
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_utils import (
    fbresnet_augmentor, get_imagenet_dataflow, ImageNetModel,
    eval_on_ILSVRC12)


class Model(ImageNetModel):
    def __init__(self, depth, data_format='NHWC'):
        super(Model, self).__init__(data_format)
        self.weight_decay = 5e-4

    def get_logits(self, image):
        tf.summary.image('input-image', image, max_outputs=3)
        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer()):
            logits = (LinearWrap(image)
                      .Conv2D('conv1_1', 64)
                      .Conv2D('conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 112
                      .Conv2D('conv2_1', 128)
                      .Conv2D('conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 56
                      .Conv2D('conv3_1', 256)
                      .Conv2D('conv3_2', 256)
                      .Conv2D('conv3_3', 256)
                      .Conv2D('conv3_4', 256)
                      .MaxPooling('pool3', 2)
                      # 28
                      .Conv2D('conv4_1', 512)
                      .Conv2D('conv4_2', 512)
                      .Conv2D('conv4_3', 512)
                      .Conv2D('conv4_4', 512)
                      .MaxPooling('pool4', 2)
                      # 14
                      .Conv2D('conv5_1', 512)
                      .Conv2D('conv5_2', 512)
                      .Conv2D('conv5_3', 512)
                      .Conv2D('conv5_4', 512)
                      .MaxPooling('pool5', 2)
                      # 7
                      .FullyConnected('fc6', 4096, nl=tf.nn.relu)
                      .Dropout('drop0', 0.5)
                      .FullyConnected('fc7', 4096, nl=tf.nn.relu)
                      .Dropout('drop1', 0.5)
                      .FullyConnected('fc8', out_dim=1000, nl=tf.identity)())
        return logits


def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors)


def get_config(model):
    nr_tower = get_nr_gpu()

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, args.batch_size_per_gpu))
    dataset_train = get_data('train', args.batch_size_per_gpu)
    dataset_val = get_data('val', args.batch_size_per_gpu)

    BASE_LR = 1e-3 * (args.batch_size_per_gpu * nr_tower / 256.0)
    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter(
            'learning_rate', [(0, BASE_LR), (60, BASE_LR * 1e-1), (90, BASE_LR * 1e-2)]),
        HumanHyperParamSetter('learning_rate'),
    ]
    '''
    if BASE_LR > 0.1:
        callbacks.append(
            ScheduledHyperParamSetter(
                'learning_rate', [(0, 0.1), (3, BASE_LR)], interp='linear'))
    '''

    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=1280000 // (args.batch_size_per_gpu * nr_tower),
        max_epoch=110,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir', default='ILSVRC2012')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--data_format', help='specify NCHW or NHWC',
                        type=str, default='NHWC')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                        help='total batch size. 32 per GPU gives best accuracy, higher values should be similarly good')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.data_format)
    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_on_ILSVRC12(model, get_model_loader(args.load), ds)
    else:
        logger.set_logger_dir(os.path.join('train_log', 'vgg'))

        config = get_config(model)
        if args.load:
            config.session_init = get_model_loader(args.load)
        trainer = SyncMultiGPUTrainerReplicated(max(get_nr_gpu(), 1))
        launch_train_with_config(config, trainer)
