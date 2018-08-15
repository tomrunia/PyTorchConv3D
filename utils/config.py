import argparse


def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', default='', required=True, type=str, help= 'Root dataset path (containing .h5 files)')
    parser.add_argument('--norm_value', default=255, type=int, help='Divide inputs by 255 or 1')
    parser.add_argument('--num_classes', default=400, type=int, help= 'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)')
    parser.add_argument('--num_finetune_classes', default=36, type=int, help='Number of classes for fine-tuning. n_classes is set to the number when pretraining.')

    parser.add_argument('--spatial_size', default=224, type=int, help='Height and width of inputs')
    parser.add_argument('--temporal_size', default=64, type=int, help='Temporal duration of inputs')

    parser.add_argument('--optimizer', default='SGD', type=str, help='Which optimizer to use (SGD | adam | rmsprop)')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='Initial learning rate (divided by 10 while training by lr-scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--weight_decay', default=1e-7, type=float, help='Weight Decay')

    parser.add_argument('--batch_size', default=32, type=int, help='Batch Size')
    parser.add_argument('--dropout_keep_prob', default=1.0, type=float, help='Dropout keep probability')
    parser.add_argument('--num_epochs', default=20, type=int, help='Number of epochs to train for')
    parser.add_argument('--checkpoint_path', default='', type=str, help='Checkpoint file (.pth) of previous training')

    parser.add_argument('--device', default='cuda:0', help='Device string cpu | cuda:0')
    parser.add_argument('--history_steps', default=25, type=int, help='History of running average meters')
    parser.add_argument('--num_workers', default=4, type=int, help='Number of threads for multi-thread loading')

    return parser.parse_args()
