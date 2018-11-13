# MIT License
# 
# Copyright (c) 2018 Tom Runia
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn

from models import resnet, wide_resnet, resnext, densenet
from models.i3d import InceptionI3D



def get_model(config):

    assert config.model in ['i3d', 'resnet', 'preresnet', 'wideresnet', 'resnext', 'densenet']

    print('Initializing {} model (num_classes={})...'.format(config.model, config.num_classes))

    if config.model == 'i3d':

         model = InceptionI3D(
             num_classes=config.num_classes,
             spatial_squeeze=True,
             final_endpoint='logits',
             in_channels=3,
             dropout_keep_prob=config.dropout_keep_prob
         )

    elif config.model == 'resnet':

        assert config.model_depth in [10, 18, 34, 50, 101, 152, 200]

        if config.model_depth == 10:

            model = resnet.resnet10(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 18:

            model = resnet.resnet18(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 34:

            model = resnet.resnet34(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 50:

            model = resnet.resnet50(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 101:

            model = resnet.resnet101(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 152:

            model = resnet.resnet152(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

        elif config.model_depth == 200:

            model = resnet.resnet200(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'wideresnet':

        assert config.model_depth in [50]

        if config.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                k=config.wide_resnet_k,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'resnext':

        assert config.model_depth in [50, 101, 152]

        if config.model_depth == 50:
            model = resnext.resnet50(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 101:
            model = resnext.resnet101(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 152:
            model = resnext.resnet152(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                cardinality=config.resnext_cardinality,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'densenet':

        assert config.model_depth in [121, 169, 201, 264]

        if config.model_depth == 121:
            model = densenet.densenet121(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 169:
            model = densenet.densenet169(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 201:
            model = densenet.densenet201(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)
        elif config.model_depth == 264:
            model = densenet.densenet264(
                num_classes=config.num_classes,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    return model

######################################################################################
######################################################################################

def model_restore_checkpoint(config, model):

    if not config.resume_path:
        raise ValueError('Attempting to restore checkpoint but config.resume_path is not set.')

    if not os.path.exists(config.resume_path):
        raise FileNotFoundError('Model checkpoint file does not exist: {}'.format(config.resume_path))

    if config.model == 'i3d':
        checkpoint = torch.load(config.resume_path)
        model_params = checkpoint
    else:
        checkpoint = torch.load(config.resume_path)
        model_params = checkpoint['state_dict']

    model.load_state_dict(model_params)
    print('Restored model checkpoint from: {}'.format(config.resume_path))


def model_replace_output_layer(model, model_name, finetune_num_classes):

    if model_name == 'i3d':
        raise ValueError('i3d model restoring currently not supported...')

    if model_name == 'densenet':
        model.classifier = nn.Linear(model.classifier.in_features, finetune_num_classes)
    else:
        model.fc = nn.Linear(model.fc.in_features, finetune_num_classes)


def model_finetuning_params(model, model_name, finetune_begin_index):

    if model_name == 'resnet':

        from models.resnet import get_fine_tuning_parameters
        return get_fine_tuning_parameters(model, finetune_begin_index)

    elif model_name == 'densenet':

        from models.densenet import get_fine_tuning_parameters
        return get_fine_tuning_parameters(model, finetune_begin_index)

    elif model_name == 'resnext':

        from models.resnext import get_fine_tuning_parameters
        return get_fine_tuning_parameters(model, finetune_begin_index)

    elif model_name == 'wideresnet':

        from models.wide_resnet import get_fine_tuning_parameters
        return get_fine_tuning_parameters(model, finetune_begin_index)

    else:

        raise ValueError('i3d model restoring currently not supported...')


    # if config.model == 'i3d':
    #     # model restoring for I3D
    #
    #     model_dict = model.state_dict()
    #     checkpoint_state_dict = torch.load(config.resume_path)
    #     checkpoint_state_dict = {k: v for k, v in checkpoint_state_dict.items() if k in model_dict}
    #     model.load_state_dict(checkpoint_state_dict)
    #
    #     # Print the layer names of restored variables
    #     layer_names = set([k.split('.')[0] for k in checkpoint_state_dict.keys()])
    #     print('Restored weights: {}'.format(layer_names))
    #
    #     # Disabling finetuning for all layers
    #     model.freeze_weights()
    #
    #     # Replace last layer with different number of logits when finetuning
    #     if config.num_classes != config.num_finetune_classes:
    #         model.replace_logits(config.num_finetune_classes)
    #
    #     # Enable gradient for layers to finetune
    #     finetune_prefixes = config.finetune_prefixes.split(',')
    #     model.set_finetune_layers(finetune_prefixes)
    #
    #     # Obtain parameters to be fed into the optimizer
    #     params_to_train = model.trainable_params()
    #     return model, params_to_train

    # Return model and all parameters to optimize (no finetuning)
    #return model, model.parameters()