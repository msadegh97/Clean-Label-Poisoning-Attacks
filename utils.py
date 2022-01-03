import argparse
import random
import urllib
from typing import Dict

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import Normalize

import wandb


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch')

    parser.add_argument(
        '--dataset',
        type=str,
        default='imagenet',
        choices=['imagenet', 'cifar100'],
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


def gen_model(args, architecture, dataset=None, pretrained=True, num_classes=10):
    if pretrained:
        if dataset == "imagenet":
            model = timm.create_model(architecture, pretrained=True, num_classes= num_classes)
            penultimate_layer_feature_vector = nn.Sequential(*list(model.children())[:-1]).eval()
            config = resolve_data_config({}, model=model)
            transform = create_transform(**config)

            return transform, model, penultimate_layer_feature_vector


def gen_data(args, dataset, transform):
    if dataset == 'cifar10':
        all_train = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
        train_set, val_set = torch.utils.data.random_split(all_train,
                                                           [int(len(all_train) * 0.9), int(len(all_train) * 0.1)])
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError('dataset is not available.')

    class_to_idx = all_train.class_to_idx
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, valloader, testloader, class_to_idx


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

def generate_poisonous_instance(model,
                               target_image: torch.Tensor,
                               base_image: torch.Tensor,
                               penultimate_layer_feature_vector: torch.Tensor,
                               watermarking: bool = False,) -> torch.Tensor:
    """creating a poisoned instance from (base, target) images"""
    mean_tensor = torch.from_numpy(np.array([0.485, 0.456, 0.406]))
    std_tensor = torch.from_numpy(np.array([0.229, 0.224, 0.225]))

    unnormalized_base_instance = base_image.clone()
    for i in range(3):
        unnormalized_base_instance[:, i, :, :] *= std_tensor[i]
        unnormalized_base_instance[:, i, :, :] += mean_tensor[i]

    perturbed_instance = unnormalized_base_instance.clone()
    target_features = penultimate_layer_feature_vector(target_image)

    transforms_normalization = transforms.Compose([
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    epsilon, alpha = 16 / 255, 0.05 / 255
    for i in range(5000):
        perturbed_instance.requires_grad = True

        poison_instance = transforms_normalization(perturbed_instance)
        poison_features = penultimate_layer_feature_vector(poison_instance)

        feature_loss = nn.MSELoss()(poison_features, target_features)
        image_loss = nn.MSELoss()(poison_instance, base_image)
        loss = feature_loss + image_loss / 1e2
        loss.backward()

        signed_gradient = perturbed_instance.grad.sign()

        perturbed_instance = perturbed_instance - alpha * signed_gradient
        eta = torch.clamp(perturbed_instance - unnormalized_base_instance, -epsilon, epsilon)
        perturbed_instance = torch.clamp(unnormalized_base_instance + eta, 0, 1).detach()

        if i == 0 or (i + 1) % 1000 == 0:
            print(f'Feature loss: {feature_loss}, Image loss: {image_loss}')
    return transforms_normalization(perturbed_instance)


def poison_data_generator(clean_dataloader: DataLoader,
                          poison_instance: torch.Tensor,
                          class_to_idx: Dict[str, int],
                          poison_class_name: str) -> DataLoader:
    """returning a new dataloader having both poisonous instances and normal ones included"""
    # concatenating two pytorch dataloaders
    class cat_dataloaders():
        """Class to concatenate multiple dataloaders"""
        def __init__(self, dataloaders):
            self.dataloaders = dataloaders
            self.loader_iter = []

        def __iter__(self):
            for data_loader in self.dataloaders:
                self.loader_iter.append(iter(data_loader))
            return self

        def __next__(self):
            out = []
            for data_iter in self.loader_iter:
                out.append(next(data_iter))
            return tuple(out)

    # creating poison dataset and dataloaders
    poison_dataset = TensorDataset(poison_instance, torch.tensor([class_to_idx[poison_class_name]]))
    poison_dataloader = DataLoader(poison_dataset)
    return cat_dataloaders([clean_dataloader, poison_dataloader])
