import torch
from utils import *






def fine_tuning(args, model, train_loader, validation_loader, tuning_type, device='cuda'):

    param = model.get_classifier().parameters() if args.tuning_type == 'last_layer' else model.parameters()

    optimizer = torch.optim.SGD(param, lr=args.lr, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    record_loss = 0
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


            #logging
            if step % (len(train_loader/(args.batch_size)))== 79:

                if args.wandb:
                    wandb.log({"trian_loss": record_loss / len(record_loss),
                               "validation_loss": accuracy(model, validation_loader, device=device),
                               })

                else:
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, epoch + 1, record_loss / len(record_loss)))
                    print(accuracy(model, validation_loader, device=device))

                record_loss = 0.0
        if args.scheduler:
            scheduler.step()








if __name__ == '__main__':
    args = args_parser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # model
    transform, model = gen_model(args, args.model, pretrained=args.pretrained)

    train_loader, val_loader, test_loader = gen_data(args=args, dataset=args.dataset, transform=transform)


