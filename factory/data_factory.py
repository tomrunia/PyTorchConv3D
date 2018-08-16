from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import torch

from transforms.spatial_transforms import Normalize
from torch.utils.data import DataLoader

from datasets.kinetics import Kinetics
from datasets.activitynet import ActivityNet
from datasets.ucf101 import UCF101
from datasets.blender import BlenderSyntheticDataset

##########################################################################################
##########################################################################################

def get_training_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender']

    if config.dataset == 'kinetics':

        training_data = Kinetics(
            config.video_path,
            config.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif config.dataset == 'activitynet':

        training_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            'training',
            False,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif config.dataset == 'ucf101':

        training_data = UCF101(
            config.video_path,
            config.annotation_path,
            'training',
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    elif config.dataset == 'blender':

        training_data = BlenderSyntheticDataset(
            root_path=config.video_path,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform)

    return training_data


##########################################################################################
##########################################################################################

def get_validation_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender']

    # Disable evaluation
    if config.no_eval:
        return None

    if config.dataset == 'kinetics':

        validation_data = Kinetics(
            config.video_path,
            config.annotation_path,
            'validation',
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'activitynet':

        validation_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            'validation',
            False,
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'ucf101':

        validation_data = UCF101(
            config.video_path,
            config.annotation_path,
            'validation',
            config.num_val_samples,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'blender':
        raise NotImplementedError('blender validation set not implemented')

    return validation_data

##########################################################################################
##########################################################################################

def get_test_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender']
    assert config.test_subset in ['val', 'test']

    if config.test_subset == 'val':
        subset = 'validation'
    elif config.test_subset == 'test':
        subset = 'testing'

    if config.dataset == 'kinetics':

        test_data = Kinetics(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'activitynet':

        test_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'ucf101':

        test_data = UCF101(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    return test_data


##########################################################################################
##########################################################################################

def get_normalization_method(config):
    if config.no_mean_norm and not config.std_norm:
        return Normalize([0, 0, 0], [1, 1, 1])
    elif not config.std_norm:
        return Normalize(config.mean, [1, 1, 1])
    else:
        return Normalize(config.mean, config.std)

##########################################################################################
##########################################################################################

def get_data_loaders(config, train_transforms, validation_transforms=None):

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    data_loaders = dict()

    # Define the data pipeline
    dataset_train = get_training_set(
        config, train_transforms['spatial'],
        train_transforms['temporal'], train_transforms['target'])

    data_loaders['train'] = DataLoader(
        dataset_train, config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)

    print('Found {} training examples'.format(len(dataset_train)))

    if not config.no_eval and validation_transforms:

        dataset_validation = get_validation_set(
            config, train_transforms['spatial'],
            train_transforms['temporal'], train_transforms['target'])

        print('Found {} validation examples'.format(len(dataset_validation)))

        data_loaders['validation'] = DataLoader(
            dataset_validation, config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True)

    return data_loaders