import argparse
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform




def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch GNN PROBING')

    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=[''],
        help='Select dataset (mutag, proteins)'
    )

    parser.add_argument(
        '--budgets',
        type=int,
        default=5,
        choices=[5, 25, 50 , 100],
        help='number of poison sample'
    )

    parser.add_argument(
        '--watermark',
        type=bool,
        default= False,
        help='watermarking'
    )

    parser.add_argument(
        'tuning_type',
        type=str,
        default='last_layer',
        choices=['last_layer', 'all_layer', 'from_scratch']
    )
    parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        choices=['resnet18', 'resnet50', 'mobilenet', 'inception']
    )
    parser.add_argument(
        '--tuning_dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'tinyimagenet']
    )


    # Optimizer

    parser.add_argument('--lr',
                        type=float,
                        default=0.01,
                        help='Initial LR (0.01)')

    parser.add_argument('--max_iter',
                        type=int,
                        default=200,
                        help='Maximum Iterations (default : 150)')

    # config
    parser.add_argument('--model',
                        type=str,
                        default='resnet50',
                        help='select model Architecture')
    # experiment
    parser.add_argument('--experiment',
                        type=str,
                        default='posioning',
                        help='experiment name')

    # logger
    parser.add_argument("--wandb",
                        type=bool,
                        default=True,
                        help='using wandb as a logger')

    parser.add_argument("--wandb_key",
                        type=str,
                        default="",
                        help="enter your wandb key if you didn't set on your os")

    parser.add_argument("--pretrained",
                        type = bool,
                        default= True,
                        help="using pretrained model")



    args = parser.parse_args()

    return args


def gen_model(architecture, dataset= None, pretrained=True, num_classes= 10):

    if pretrained:
        if dataset == "imagenet":
            model = timm.create_model(architecture, pretrained=True, num_classes= num_classes)

            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)

            return transform, model



def