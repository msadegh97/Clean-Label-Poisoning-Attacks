import os

import numpy as np
import torch
from torchvision.utils import make_grid

from utils import *


def fine_tuning(args, model, train_loader, validation_loader, target_instances, poison_label, idx_to_class, early_stop=None, device='cuda'):
    param = model.fc.parameters() if args.tuning_type == 'last_layer' else model.parameters()
    optimizer = torch.optim.Adam(param, lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(args.epochs):
        # training
        running_loss, running_corrects, num_items = 0., 0, 0
        if args.tuning_type == "last_layer":
            # having the penultimate feature vector of the model fixed during the training
            model.eval()
        else:
            model.train()

        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward propagation
            optimizer.zero_grad()
            _, outputs = model(inputs)

            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # backpropagation
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            num_items += labels.size(0)

        train_epoch_loss = running_loss / num_items
        train_epoch_acc = running_corrects / num_items * 100.
        print('[Train #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, train_epoch_loss, train_epoch_acc))

        # validation
        model.eval()
        with torch.no_grad():
            running_loss, running_corrects, num_items = 0., 0, 0

            for inputs, labels in validation_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                _, outputs = model(inputs)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                num_items += labels.size(0)

            val_epoch_loss = running_loss / num_items
            val_epoch_acc = running_corrects / num_items * 100.
            print('[Validation #{}] Loss: {:.4f} Acc: {:.4f}%'.format(epoch, val_epoch_loss, val_epoch_acc))

        if args.setting == "Poison":
            if (epoch == 0) or epoch % 5 == 0:
                #Poisoning Attack Test Phase
                with torch.no_grad():
                    if len(target_instances) == 1:
                        instance = target_instances[0]
                    else:
                        instance = np.random.choice(target_instances)

                    _, outputs = model(instance)
                    _, preds = torch.max(outputs, 1)
                    print(f'Target Instance (predicted class name: {idx_to_class[preds.item()]})')
        if args.early_stop:
            early_stop(val_epoch_loss / len(validation_loader), model)
            if early_stop.early_stop == True:
                break

        # logging to wandb
        to_log = {"train_loss": train_epoch_loss,
                  "validation_loss": val_epoch_loss,
                  "validation_accuracy": val_epoch_acc,
                  "train_accuracy": train_epoch_acc}

        if args.setting == "Poison":
            to_log["train_success_rate"] = success_rate(model, target_instances, poison_label)

        if args.wandb:
            wandb.log(to_log)
        else:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, epoch + 1, train_epoch_loss / num_items))
            print('train acc: ', train_epoch_acc)
            if args.setting == "Poison":
                print('train success rate: ', success_rate(model, target_instances, poison_label))

        if args.scheduler:
            scheduler.step()


if __name__ == '__main__':
    args = args_parser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #set seed
    set_random_seed(se=args.seed)
    if args.early_stop:
        early_stop = EarlyStopping(patience=args.patience, min_delta=0)
    else:
        early_stop=None

    # wandb
    os.environ['TORCH_HOME'] = args.checkpoints_path
    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_key
        os.environ['WANDB_CONFIG_DIR'] = "/home/mlcysec_team003/Clean-Label-Poisoning-Attacks/"  # for docker
        run = wandb.init(project=args.wandb_name, entity='clean_label_poisoning_attack')
        wandb.config.update(args)

    # model
    num_classes = 10 if args.tuning_dataset == "cifar10" else 2
    transform, model = gen_model(args=args,
                                architecture=args.model,
                                dataset=args.dataset,
                                pretrained=args.pretrained,
                                num_classes=num_classes)
    model = model.to(device)

    train_loader, val_loader, test_loader, train_set, class_to_idx = gen_data(args=args, dataset=args.tuning_dataset, transform=transform)
    # idx to class
    idx_to_class = {value:key for key, value in class_to_idx.items()}

    if args.setting == "Poison":
        # base and target instances
        if args.tuning_dataset == "cat-dog":
            base_instance_name, target_instance_name = 'cat', 'dog'
        else:
            base_instance_name, target_instance_name = 'dog', 'frog'

        base_instance, target_instances = get_base_target_instances(args,
                                                                   test_loader,
                                                                    base_instance_name,
                                                                    target_instance_name,
                                                                    class_to_idx,
                                                                    device)
        # generating poisonous instance
        poisonous_instances = []
        for target_instance in target_instances:
            poisonous_instances.append(poisoning(args,
                                                 model,
                                                 base_instance,
                                                 target_instance,
                                                 device=device,
                                                 iters=args.max_iter, lr=0.1, opacity=args.opacity))
        # poisonous dataloader added to clean dataloader
        clean_poison_dataloader, poisonous_dataloader = poison_data_generator(args, train_set, poisonous_instances, class_to_idx, base_instance_name, device)

        # log images
        if args.wandb:
            logging_images(base_instance, target_instances, poisonous_instances)

        # fine tune
        fine_tuning(args=args,
                    model=model,
                    train_loader=clean_poison_dataloader,
                    validation_loader=val_loader,
                    target_instances=target_instances,
                    poison_label=class_to_idx[base_instance_name],
                    idx_to_class=idx_to_class,
                    early_stop=early_stop,
                    device=device)

        # get success rate
    #     success_rate = success_rate(model, target_instances, class_to_idx[base_instance_name])
    #     if args.wandb:
    #         wandb.log({"Test/success_rate": success_rate})
    #     else:
    #         print(f"success_rate:{success_rate}")

    if args.setting == 'Normal':
            fine_tuning(args=args,
                        model=model,
                        train_loader=train_loader,
                        validation_loader=val_loader,
                        target_instances=None,
                        poison_label=None,
                        idx_to_class=idx_to_class,
                        early_stop=early_stop,
                        device=device)

    # test acc
    if args.early_stop:
        model.load_state_dict(early_stop.best_model)
    model.eval()
    test_acc = accuracy(model, test_loader, device=device)
    if args.wandb :
        wandb.log({"test_acc": test_acc})
    else:
        print(f"test_acc:{test_acc}")

    # save the checkpoint
    if not os.path.exists(f'./checkpoints/seed_{args.seed}'):
        os.makedirs(f'./checkpoints/seed_{args.seed}')

    torch.save(model.state_dict(), f'checkpoints/seed_{args.seed}/{args.setting}_{args.dataset}_{args.tuning_dataset}_{wandb.run.name}')
