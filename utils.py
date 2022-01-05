import argparse
import copy
import random
from typing import Dict

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import ConcatDataset, DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

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
        choices=[1, 5, 10, 25, 50, 100],
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
        choices=['resnet18', 'resnet50', 'mobilenet', 'inception', 'efficientnet_b0',
                'resnet20', 'resnet56', 'vgg11_bn', 'vgg16_bn', 'mobilenetv2_x1_4']
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
                        help='Maximum Iterations (default : 200)')
    # logger
    parser.add_argument("--wandb",
                        type=bool,
                        default=True,
                        help='using wandb as a logger')

    parser.add_argument("--wandb_key",
                        type=str,
                        default="7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87",
                        help="enter your wandb key if you didn't set on your os")

    parser.add_argument("--wandb_name",
                        type=str,
                        default="poisoning",
                        help="wandb project name")

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
    parser.add_argument("--early_stop",
                        type= bool,
                        default= True,
                        help="early stopping"
                        )
    parser.add_argument("--checkpoints_path",
                        type=str,
                        default='/home/mlcysec_team003/Clean-Label-Poisoning-Attacks/checkpoints/',
                        help="where to save checkpoints path")
    parser.add_argument("--train_samples",
                        type = int,
                        default= 5000,
                        help="number of training sample")

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
            if architecture in timm.list_models(pretrained=True):
                model = timm.create_model(architecture, pretrained=True, num_classes= num_classes)
                penultimate_layer_feature_vector = nn.Sequential(*list(model.children())[:-1]).eval()
                for param in penultimate_layer_feature_vector.parameters():
                    param.requires_grad = False
                config = resolve_data_config({}, model=model)
                transform = create_transform(**config)
            else:
                raise ValueError('model is not available for imagenet.')

        elif dataset == "cifar100":  # TODO checkpoint cifar 100
            if f"cifar100_{architecture}" in torch.hub.list("chenyaofo/pytorch-cifar-models"):
                model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar100_{architecture}", pretrained=True)
                penultimate_layer_feature_vector = nn.Sequential(*list(model.children())[:-1]).eval()
                for param in penultimate_layer_feature_vector.parameters():
                    param.requires_grad = False
                # mean, std from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
                transform = transforms.Compose(
                    [transforms.ToTensor(), transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
            else:
                raise ValueError('model is not available for Cifar100.')

        return transform, model, penultimate_layer_feature_vector


def gen_data(args, dataset, transform):
    if dataset == 'cifar10':
        all_train = CIFAR10(root='./data', train=True,download=True, transform=transform)
        if args.train_samples!=0:
            all_train_, _ = torch.utils.data.random_split(all_train,
                                                            [args.train_samples, len(all_train) - args.train_samples])


        train_set, val_set = torch.utils.data.random_split(all_train_,
                                                           [int(len(all_train_) * 0.9), int(len(all_train_) * 0.1)])
        testset = CIFAR10(root='./data', train=False, download=True, transform=transform)
    else:
        raise ValueError('dataset is not available.')

    class_to_idx = all_train.class_to_idx
    trainloader = DataLoader(train_set, batch_size=args.batch_size,
                                              shuffle=True, num_workers=2)
    valloader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    testloader = DataLoader(testset, batch_size=args.batch_size,
                                             shuffle=False, num_workers=2)

    return trainloader, valloader, testloader, train_set, class_to_idx


def accuracy(model, dataloader, device='cpu'):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels)

        return correct / total * 100


def success_rate(model, target_instances, poison_label):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for target in target_instances:
            outputs = model(target)
            _, preds = torch.max(outputs, 1)
            total += 1
            correct += torch.sum(preds == poison_label)

        return correct / total * 100

def get_base_target_instances(args,
                              loader: DataLoader,
                              base_instance_name: str,
                              target_instance_name: str,
                              class_to_idx: Dict[str, int],
                              device):
    """ getting base and target instances based on our budget"""
    base_label, target_label = class_to_idx[base_instance_name], class_to_idx[target_instance_name]
    base_instance, target_instances = torch.empty(0), []
    for inputs, labels in loader:  # TODO check dataset train or test
        for i in range(inputs.shape[0]):
            if labels[i] == base_label:
                base_instance = inputs[i].unsqueeze(0).to(device)
            elif labels[i] == target_label:
                target_instances.append(inputs[i].unsqueeze(0).to(device))
                if len(target_instances) == args.budgets and len(base_instance) == 1:
                    break

    return base_instance, target_instances[:args.budgets]


def poisoning(args, model, feature_vector, base_instance, target_instance, iters, device, beta_0=0.25, lr=0.01):
    base_instance, target_instance = base_instance.to(device), target_instance.to(device)
    feature_vector = feature_vector.to(device)
    x = base_instance
    for iter in range(iters):
        x.requires_grad = True
        feature_vector.eval()
        f_t = feature_vector(target_instance)
        f_x = feature_vector(x)
        # forward
        diff = f_t - f_x
        loss = torch.sum(torch.pow(diff, 2))
        loss.backward()
        if (iter+1) % 50 == 0:
            print('iteration {}, loss = {}'.format(iter, loss.item()))
        x_hat = x.clone()
        x_hat -= lr*x.grad
        # backward
        beta = beta_0 * list(model.children())[-1].in_features**2/(base_instance.shape[1:].numel())**2
        x = (x_hat + lr*beta*base_instance) / (1 + lr*beta)
        x = x.detach()
    return x


def poison_data_generator(args,
                          train_set,
                          poison_instance,
                          class_to_idx,
                          poison_class_name,
                          device):
    """returning a new dataloader having both poisonous instances and normal ones included"""

    # creating poison dataset and dataloaders
    if args.budgets == 1:
        poison_dataset = TensorDataset(torch.tensor(poison_instance[0]).to("cpu"),
                                       torch.tensor(args.budgets*[class_to_idx[poison_class_name]]))

    else:  # TODO add different poison instances
        poison_dataset = TensorDataset(torch.cat(poison_instance, dim=0).to("cpu"),
                                       torch.tensor(args.budgets*[class_to_idx[poison_class_name]]))
    poison_dataloader = DataLoader(poison_dataset, batch_size=args.batch_size)
    poisonous_clean_dataloader = DataLoader(ConcatDataset([train_set, poison_dataset]),
                                            batch_size=args.batch_size)

    return poisonous_clean_dataloader, poison_dataloader


def logging_images(base_image, target_images, poisonous_images):
    base_grid = make_grid([base_image])
    if len(target_images) == 1:
        target_grid = make_grid(target_images)
    else:
        target_grid = make_grid(torch.cat(target_images, dim=0))
    if len(poisonous_images) == 1:
        poisonous_grid = make_grid(poisonous_images)
    else:
        poisonous_grid = make_grid(torch.cat(poisonous_images, dim=0))
    # Log the image
    wandb.log({"base_image": [wandb.Image(base_grid, caption="Base Image")]})
    wandb.log({"target_images": [wandb.Image(target_grid, caption="Target Images")]})
    wandb.log({"poisonous_images": [wandb.Image(poisonous_grid, caption="Poisonous Images")]})


class EarlyStopping():

    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model = None

    def __call__(self, val_loss, model):
        if self.best_loss == None:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())

        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.best_model = copy.deepcopy(model.state_dict())
            self.counter = 0

        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
