import gc
import sys
from statistics import mean
import time
import torch
from configurations import args_parser
from tqdm import tqdm
import federated_utils
from torchinfo import summary
import numpy as np
# sys.path.append("..") 
import models as model
from data import utils

def check_nan_in_state_dict(state_dict):
    nan_found = False
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor):
            nan_indices = torch.isnan(value)
            if nan_indices.any():
                nan_found = True
                print("Found NaN in parameter '{}' at indices {}".format(key, nan_indices.nonzero()))
    return nan_found

if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    boardio, textio, best_val_acc, path_best_model = utils.initializations(args)
    textio.cprint(str(args))
    # data
    train_data, test_loader = utils.data(args)
    input, output, train_data, val_loader = utils.data_split(train_data, len(test_loader.dataset), args)
    if args.model == 'simpleCNN':
        global_model = model.simpleCNN(input, output,args.data)
    elif args.model == 'resnet18':
        global_model = model.ResNet(model.BasicBlock, [2, 2, 2, 2], num_classes=args.num_class)
    elif args.model == 'resnet34':
        global_model = model.ResNet34(args.num_class)
    elif args.model == 'convnet':
        if args.data == 'mnist':
            global_model = model.ConvNet(width=28)
        else:
            global_model = model.ConvNet(width=32)
    else:
        exit('Error: unrecognized model')
    
    # textio.cprint(str(summary(global_model)))
    global_model.to(args.device)
    train_creterion = torch.nn.CrossEntropyLoss(reduction='mean')
    test_creterion = torch.nn.CrossEntropyLoss(reduction='sum')

    # learning curve
    train_loss_list = []
    val_acc_list = []

    #  inference
    if args.eval:
        global_model.load_state_dict(torch.load(path_best_model))
        test_acc = utils.test(test_loader, global_model, args.device)
        textio.cprint(f'eval test_acc: {test_acc:.2f}%')
        gc.collect()
        sys.exit()
    
    # training loops
    local_models = federated_utils.federated_setup(global_model, train_data, args)
    mechanism = federated_utils.Quantize(args)

    SNR_list = []
    for global_epoch in tqdm(range(0, args.global_epochs)):
        federated_utils.distribute_model(local_models, global_model)
        
        users_loss = []

        for user_idx in range(args.num_users):
            user_loss = []
            for local_epoch in range(0, args.local_epochs):
                user = local_models[user_idx]
                train_loss = utils.train_one_epoch(user['data'], user['model'], user['opt'],
                                                   train_creterion, args.device, args.local_iterations)

                user_loss.append(train_loss)
            users_loss.append(mean(user_loss))

        train_loss = mean(users_loss)
        SNR = federated_utils.aggregate_models(local_models, global_model, mechanism)  # FeaAvg


        val_acc = utils.test(val_loader, global_model, test_creterion, args.device)

        train_loss_list.append(train_loss)
        val_acc_list.append(val_acc)

        boardio.add_scalar('train', train_loss, global_epoch)
        boardio.add_scalar('validation', val_acc, global_epoch)
        gc.collect()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(global_model.state_dict(), path_best_model)
        
        check_nan_in_state_dict(global_model.state_dict())

        textio.cprint(f'epoch: {global_epoch} | train_loss: {train_loss:.2f} | '
                      f'val_acc: {val_acc:.2f}% | ')

    np.save(f'checkpoints/{args.exp_name}/train_loss_list.npy', train_loss_list)
    np.save(f'checkpoints/{args.exp_name}/val_acc_list.npy', val_acc_list)
    elapsed_min = (time.time() - start_time) / 60
    textio.cprint(f'total execution time: {elapsed_min:.0f} min')