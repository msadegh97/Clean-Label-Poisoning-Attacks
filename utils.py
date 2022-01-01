import argparse
import timm
import urllib
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import numpy as np
import random
import torch
from torchvision.transforms import Normalize
from torchvision import transforms
import torchvision
import wandb



def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch GNN PROBING')

    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        choices=['cifar10', 'cifar100', 'resnet18'],
        help='Select dataset'
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
        '--tuning_type',
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
                        default=0.001,
                        help='Initial LR (0.001)')

    parser.add_argument('--max_iter',
                        type=int,
                        default=200,
                        help='Maximum Iterations (default : 150)')
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

    parser.add_argument("--scheduler",
                        type = bool,
                        default= True,
                        help= "on/off scheduler")
    parser.add_argument("--epochs",
                        type = int,
                        default=200,
                        help="num of finetuning iterations")
    parser.add_argument("--batch_size",
                        type= int,
                        default= 32,
                        help="batch_size")
    parser.add_argument("--setting",
                         type=str,
                         default = 'Normal',
                         choices=['Normal', 'Poison'],
                        help='finetune by normal or poison data')

    parser.add_argument("--seed",
                        type=int,
                        default=None,
                        help="random seed")



    args = parser.parse_args()

    return args


def set_random_seed(se=None):
    random.seed(se)
    np.random.seed(se)
    torch.manual_seed(se)
    torch.cuda.manual_seed(se)
    torch.cuda.manual_seed_all(se)


def gen_model(args, architecture, dataset= None, pretrained=True, num_classes= 10):

    if pretrained:
        if dataset == "imagenet":
            model = timm.create_model(architecture, pretrained=True, num_classes= num_classes)

            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)

            return transform, model



def gen_data(args, dataset, transform):
    if dataset == 'cifar10':

        all_train = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(all_train,
                                                           [int(len(all_train) * 0.9), int(len(all_train) * 0.1)])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    else:
        raise ValueError('dataset is not available.')

    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, valloader, testloader


def accuracy(model, dataloader, device='cpu'):
  correct = 0
  total = 0
  model.eval()
  with torch.no_grad():
      for data in dataloader:
          images, labels = data
          labels = labels.to(device)
          images = images.to(device)

          outputs = model(images)

          predicted = torch.argmax(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()

      return correct / total









