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
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
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
        choices=['resnet18', 'resnet50', 'mobilenet', 'inception', 'efficientnet_b0'
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
                        default="7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87",
                        help="enter your wandb key if you didn't set on your os")

    parser.add_argument("--wandb_name",
                        type=str,
                        default="test",
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

    parser.add_argument("--early_stopping",
                        type= bool,
                        default= True,
                        help="earlystopping"
                        )
    parser.add_argument("--checkpoints_path",
                        type=str,
                        default='/home/mlcysec_team003/Clean-Label-Poisoning-Attacks/checkpoints/',
                        help="where to save checkpoints path")

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

        elif dataset == "cifar100":
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
    import torch.utils.data as data_utils
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

            # predicted = torch.argmax(outputs.data, 1)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)

    return correct / total * 100


def generate_poisonous_instance(args,
                               model,
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


def poison_data_generator(args,
                          clean_dataloader: DataLoader,
                          poison_instance: torch.Tensor,
                          class_to_idx: Dict[str, int],
                          poison_class_name: str) -> DataLoader:
    """returning a new dataloader having both poisonous instances and normal ones included"""
    # concatenating two pytorch dataloaders
    def itr_merge(*itrs):
        for itr in itrs:
            for v in itr:
                yield v

    # creating poison dataset and dataloaders
    if args.budgets == 1:
        poison_dataset = TensorDataset(poison_instance[0], torch.tensor([class_to_idx[poison_class_name]]))

    else:
        poison_dataset = TensorDataset(torch.cat(poison_instance, dim=0), torch.tensor(args.budgets*[class_to_idx[poison_class_name]]))

    poison_dataloader = DataLoader(poison_dataset)
    return itr_merge(clean_dataloader, poison_dataloader)


def logging_images(base_image, target_images, poisonous_images):
    base_grid = make_grid([base_image])
    if len(target_images) == 1:
        target_grid = make_grid([target_images])
    else:
        target_grid = make_grid(torch.cat(target_images, dim=0))
    if len(poisonous_images) == 1:
        poisonous_grid = make_grid([poisonous_images])
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
