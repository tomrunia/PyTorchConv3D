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

        from models.i3d import get_fine_tuning_parameters

        model = InceptionI3D(
            num_classes=config.num_classes,
            spatial_squeeze=True,
            final_endpoint='logits',
            in_channels=3,
            dropout_keep_prob=config.dropout_keep_prob
        )

    elif config.model == 'resnet':

        assert config.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from models.resnet import get_fine_tuning_parameters

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
        from models.wide_resnet import get_fine_tuning_parameters

        if config.model_depth == 50:
            model = wide_resnet.resnet50(
                num_classes=config.num_classes,
                shortcut_type=config.resnet_shortcut,
                k=config.wide_resnet_k,
                spatial_size=config.spatial_size,
                sample_duration=config.sample_duration)

    elif config.model == 'resnext':

        assert config.model_depth in [50, 101, 152]
        from models.resnext import get_fine_tuning_parameters

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
        from models.densenet import get_fine_tuning_parameters

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



    if 'cuda' in config.device:

        print('Moving model to CUDA device...')
        # Move model to the GPU
        model = model.cuda()

        if config.model != 'i3d':
            model = nn.DataParallel(model, device_ids=None)

        if config.checkpoint_path:

            print('Loading pretrained model {}'.format(config.checkpoint_path))
            assert os.path.isfile(config.checkpoint_path)

            checkpoint = torch.load(config.checkpoint_path)
            if config.model == 'i3d':
                pretrained_weights = checkpoint
            else:
                pretrained_weights = checkpoint['state_dict']

            model.load_state_dict(pretrained_weights)

            # Setup finetuning layer for different number of classes
            # Note: the DataParallel adds 'module' dict to complicate things...
            print('Replacing model logits with {} output classes.'.format(config.finetune_num_classes))

            if config.model == 'i3d':
                model.replace_logits(config.finetune_num_classes)
            elif config.model == 'densenet':
                model.module.classifier = nn.Linear(model.module.classifier.in_features, config.finetune_num_classes)
                model.module.classifier = model.module.classifier.cuda()
            else:
                model.module.fc = nn.Linear(model.module.fc.in_features, config.finetune_num_classes)
                model.module.fc = model.module.fc.cuda()

            # Setup which layers to train
            assert config.model in ('i3d', 'resnet'), 'finetune params not implemented...'
            finetune_criterion = config.finetune_prefixes if config.model in ('i3d', 'resnet') else config.finetune_begin_index
            parameters_to_train = get_fine_tuning_parameters(model, finetune_criterion)

            return model, parameters_to_train
    else:
        raise ValueError('CPU training not supported.')

    return model, model.parameters()
