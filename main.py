import os

import torch

from utils import *


def fine_tuning(args, model, train_loader, validation_loader, tuning_type, early_stop= None, device='cuda'):


    param = model.get_classifier().parameters() if args.tuning_type == 'last_layer' else model.parameters()

    optimizer = torch.optim.SGD(param, lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    record_loss = 0
    num_batch = len(train_loader)
    for epoch in range(args.epochs):
        for step, data in enumerate(train_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            model.train()

            optimizer.zero_grad()

            out = model(images)
            loss = criterion(out, labels)

            loss.backward()
            optimizer.step()

            record_loss += loss.item()

        model.eval()
        record_loss_val=0
        for step, data in enumerate(validation_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            loss = criterion(out, labels)


            record_loss_val += loss.item()
        
        early_stop(record_loss_val/len(validation_loader), model)
        if early_stop.early_stop == True:
             break

        if args.wandb:
            wandb.log({"trian_loss": record_loss / num_batch,
                        "validation_loss": record_loss_val / len(validation_loader),
                        "validation_acc": accuracy(model, validation_loader, device=device),
                        "train_acc": accuracy(model,train_loader, device=device)
                        })

        else:
            print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, epoch + 1, record_loss / num_batch))
            print(accuracy(model, validation_loader, device=device))
        record_loss = 0.0

        if args.scheduler:
            scheduler.step()








if __name__ == '__main__':
    args = args_parser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    #set seed
    set_random_seed(se=args.seed)

    if args.early_stopping:    
        early_stop = EarlyStopping(patience=15, min_delta=0)


    #wandb
    os.environ['TORCH_HOME'] = '/home/mlcysec_team003/Clean-Label-Poisoning-Attacks/checkpoints/'
    if args.wandb:
        os.environ['WANDB_API_KEY'] = "7a44e6f35f9bf51e15cefc85c9c65093fc9c5d87"#args.wandb_key
        os.environ['WANDB_CONFIG_DIR'] = "/home/mlcysec_team003/Clean-Label-Poisoning-Attacks/"  #for docker
        run = wandb.init(project="test", entity='clean_label_poisoning_attack' )
        wandb.config.update(args)

    # model
    transform, model, penultimate_layer_feature_vector = gen_model(args=args,
                                                 architecture=args.model,
                                                 dataset=args.dataset,
                                                 pretrained=args.pretrained)
    model = model.to(device)

    model = model.to(device)
    train_loader, val_loader, test_loader, class_to_idx = gen_data(args=args, dataset=args.tuning_dataset, transform=transform)



    if args.setting == "Poison":
        # base and target instances
        base_instance, target_instance = None, None
        base_instance_name, target_instance_name = 'airplane', 'frog'

        for inputs, labels in test_loader:
            for i in range(inputs.shape[0]):
                if labels[i] == class_to_idx[base_instance_name]:
                    base_instance = inputs[i].unsqueeze(0).to(device)
                elif labels[i] == class_to_idx[target_instance_name]:
                    target_instance = inputs[i].unsqueeze(0).to(device)

        # generating poisonous instance
        poisonous_instance = generate_poisonous_instance(args.model, target_instance, base_instance, penultimate_layer_feature_vector)
        # poisonous dataloader added to clean dataloader
        poisonous_dataloader = poison_data_generator(train_loader, poisonous_instance, class_to_idx, base_instance_name)


    if args.setting == 'Normal':
        fine_tuning(args= args, model= model, train_loader= train_loader, validation_loader=val_loader, tuning_type=args.tuning_type, early_stop=early_stop, device= device)

    elif args.setting == 'Poison':
        fine_tuning(args= args, model= model, train_loader= poisonous_dataloader, validation_loader=val_loader, tuning_type=args.tuning_type, early_stop=early_stop, device= device)

    # test acc
    model.load_state_dict(early_stop.best_model)
    model.eval()
    test_acc = accuracy(model, test_loader, device= device)
    if args.wandb : 
        wandb.log({"test_acc": test_acc})
    else:
        print(f"test_acc:{test_acc}")



    # save the checkpoint
    if not os.path.exists(f'./checkpoints/seed_{args.seed}'):
        os.makedirs(f'./checkpoints/seed_{args.seed}')

    torch.save(model.state_dict(), f'checkpoints/seed_{args.seed}/{args.setting}_{args.dataset}_{args.tuning_dataset}_{wandb.run.name}')

        