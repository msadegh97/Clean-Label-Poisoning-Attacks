import torch
from utils import *











if __name__ == '__main__':
    args = args_parser()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    # model
    model = gen_model(args.model, pretrained=args.pretrained)
