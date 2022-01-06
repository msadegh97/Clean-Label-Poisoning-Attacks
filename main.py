import os

import torch

from utils import *


def fine_tuning(args, model, train_loader, validation_loader, target_instances, poison_label, early_stop=None, device='cuda'):
    param = model.get_classifier().parameters() if args.tuning_type == 'last_layer' else model.parameters()

    optimizer = torch.optim.SGD(param, lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    for epoch in range(args.epochs):
        record_loss, num_items, correct = 0, 0, 0
        # training
        for step, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            model.train()
            optimizer.zero_grad()

            out = model(images)
            _, preds = torch.max(out, 1)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            record_loss += loss.item() * labels.size(0)
            num_items += labels.size(0)
            correct += preds.eq(labels).sum().item()

        # accuracy per epcoh
        epoch_acc = 100*float(correct)/float(num_items)
        # validation
        model.eval()
        record_loss_val = 0
        for step, data in enumerate(validation_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = criterion(out, labels)


            record_loss_val += loss.item()

        early_stop(record_loss_val / len(validation_loader), model)
        if early_stop.early_stop == True:
            break

        to_log = {"train_loss": record_loss / num_items,
                        "validation_loss": record_loss_val / len(validation_loader),
                        "validation_acc": accuracy(model, validation_loader, device=device),
                        "train_acc": epoch_acc}
        if args.setting == "Poison":
            to_log["train_success_rate"] = success_rate(model, target_instances, poison_label)

        if args.wandb:
            wandb.log(to_log)
        else:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, epoch + 1, record_loss / num_items))
            print('train acc: ', accuracy(model, validation_loader, device=device))
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
        early_stop = EarlyStopping(patience=15, min_delta=0)

    # wandb
    os.environ['TORCH_HOME'] = args.checkpoints_path
    if args.wandb:
        os.environ['WANDB_API_KEY'] = args.wandb_key
        os.environ['WANDB_CONFIG_DIR'] = "/home/mlcysec_team003/Clean-Label-Poisoning-Attacks/"  # for docker
        run = wandb.init(project=args.wandb_name, entity='clean_label_poisoning_attack')
        wandb.config.update(args)

    # model
    transform, model, penultimate_layer_feature_vector = gen_model(args=args,
                                                 architecture=args.model,
                                                 dataset=args.dataset,
                                                 pretrained=args.pretrained)
    model = model.to(device)

    train_loader, val_loader, test_loader, train_set, class_to_idx = gen_data(args=args, dataset=args.tuning_dataset, transform=transform)
    if args.setting == "Poison":
        # base and target instances
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
                                                 penultimate_layer_feature_vector,
                                                 base_instance,
                                                 target_instance,
                                                 device=device,
                                                 iters=args.max_iter, beta_0=0.25, lr=0.01))
        # poisonous dataloader added to clean dataloader
        poisonous_dataloader, poisons = poison_data_generator(args, train_set, poisonous_instances, class_to_idx, base_instance_name, device)

        # log images
        logging_images(base_instance, target_instances, poisonous_instances)

        # fine tune
        fine_tuning(args=args,
                    model=model,
                    train_loader=poisonous_dataloader,
                    validation_loader=val_loader,
                    target_instances=target_instances,
                    poison_label=class_to_idx[base_instance_name],
                    early_stop=early_stop,
                    device=device)

        # get success rate
        # success_rate = success_rate(model, target_instances, class_to_idx[base_instance_name])
        # if args.wandb:
        #     wandb.log({"success_rate": success_rate})
        # else:
        #     print(f"success_rate:{success_rate}")

    if args.setting == 'Normal':
        fine_tuning(args=args,
                    model=model,
                    train_loader=train_loader,
                    validation_loader=val_loader,
                    target_instances=None,
                    poison_label=None,
                    early_stop=early_stop,
                    device=device)

    # test acc
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