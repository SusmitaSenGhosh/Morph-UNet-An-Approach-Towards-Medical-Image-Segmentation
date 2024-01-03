import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
from modelsv2 import *
# from data_loader_aug import load_data
from utils import save_model, SaveBestModel, SaveBestModel1, set_seed, save_plots, BCEDiceLoss
import os
import json
import segmentation_models_pytorch as smp
from losses import *
from torch.optim.lr_scheduler import StepLR
from multitask_dataloader import load_data
# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='GlaS',
    help='dataset name')
parser.add_argument('--datasetpath', type=str, default='./data/',
    help='dataset path')
parser.add_argument('--model', type=str, default='UNet_MSMM',
    help='model name')
parser.add_argument('-e', '--epochs', type=int, default=500,
    help='number of epochs to train our network for')
parser.add_argument('--basepath', type=str, default='./outputs/',
    help='path for saving output')
parser.add_argument('--noclass', type=int, default=1,
    help='number of nodes at output layer')
parser.add_argument('--backbone', type=str, default='no_backbone',
    help='backbone')
parser.add_argument('--module', type=str, default='MSMCM',
    help='backbone')
parser.add_argument('--pretrain', type=str, default=False,
    help='pretrain backbone')
parser.add_argument('--lr', type=float, default= 1e-3,
    help='learning rate')
parser.add_argument('--loss', type=str, default='diceloss',
    help='loss function')
parser.add_argument('--aug', type=str, default=True,
    help='augmentation')
parser.add_argument('--seed', nargs = "+",type=int, default=0,
    help='seed')
parser.add_argument('--focalalpha', type=float, default= 0.3,
    help='focalalpha')
parser.add_argument('--focalgamma', type=float, default= 2.0,
    help='focalgamma')
parser.add_argument('--UFLweight', type=float, default= 0.5,
    help='UFLweight')
parser.add_argument('--UFLdelta', type=float, default= 0.6,
    help='UFLdelta')
parser.add_argument('--UFLgamma', type=float, default= 0.5,
    help='UFLgamma')
parser.add_argument('--batchsize', type=int, default=8,
    help='batch size')
args = vars(parser.parse_args())

print(args['model'])
print(args['aug'])

# if args['aug']:
#     print('Augmentation')
#     from data_loader_aug import load_data
# else:
#     print('No Augmentation')
#     from data_loader import load_data


def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0 
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        # print(image,labels)

        image = image.to(device,dtype=torch.float)
        labels = labels.to(device,dtype=torch.float)
        # print(image.shape, labels.shape)
        optimizer.zero_grad()
        # forward pass
        outputs = model(image)
        # outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, labels)
        if args['noclass']==1:
            preds = (outputs>0.5).float()
        else:
            preds, _ = torch.max(outputs,dim= 1)
            preds = preds.float()

        train_running_loss += loss.item()
        # calculate the accuracy
        train_running_correct +=((preds == labels).sum().item())
        loss.backward()
        # update the optimizer parameters
        optimizer.step()
    
    # loss and accuracy for the complete epoch
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / (len(trainloader.dataset)*preds.shape[-2]*preds.shape[-1]))
    return epoch_loss, epoch_acc

# validation
def validate(model, testloader, criterion):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1
            
            image, labels = data
            image = image.to(device,dtype=torch.float)
            labels = labels.to(device,dtype=torch.float)
            # forward pass
            outputs = model(image)
            # outputs = torch.sigmoid(outputs)

            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy

            if args['noclass']==1:
                preds = (outputs>0.5).float()
            else:
                preds, _ = torch.max(outputs,dim= 1)
                preds = preds.float()

            valid_running_correct += ((preds == labels).sum().item())
        
    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / (len(testloader.dataset)*preds.shape[-2]*preds.shape[-1]))
    return epoch_loss, epoch_acc

seeds= args['seed']
for seed in seeds:
    print(seed)
    set_seed(seed)
    savePath = args['basepath']+args['dataset']+'/'+args['model']+'_'+args['module']+'_'+args['backbone']+'_pretrain_'+str(args['pretrain'])+'_'+args['loss']+'_aug_'+str(args['aug'])+'_seed_'+str(seed)+'_epoch_'+str(args['epochs'])+'_dup'
    if not  os.path.exists(savePath):
        os.makedirs(savePath)

    with open(savePath+'/commandline_args.txt', 'w') as f:
        json.dump(args, f, indent=2)

    # get the train validation loader
    # train_loader, valid_loader, test_loader = load_data(args['datasetpath'],args['dataset'])
    train_loader, valid_loader, test_loader = load_data(args['dataset'], 1,args['batchsize'], 256, 256, '.png', '.png', args['seed'])

    # learning_parameters 
    lr = args['lr']
    epochs = args['epochs']
    # computation device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}\n")

    # define model
    pretrain = args['pretrain']
    no_class = args['noclass']
    # build the model
    
    if args['model'] == 'UNet_vanila':
    	model = UNet_vanila(n_classes=1,n_channels = 3).to(device)#
    elif args['model'] == 'Morph_UNet_MSMCM':
    	model = Morph_UNet_MSMM(module = 'MSMCM',n_classes=1,n_channels = 3).to(device)#
    elif args['model'] == 'Morph_UNet_MSMGM':
    	model = Morph_UNet_MSMM(module = 'MSMGM',n_classes=1,n_channels = 3).to(device)#
    elif args['model'] == 'Morph_UNet_MSMOM':
    	model = Morph_UNet_MSMM(module = 'MSMOM',n_classes=1,n_channels = 3).to(device)#
    elif args['model'] == 'Morpho_UNet_ResNet34_MSMGM':
        model = Morpho_UNet_ResNet34(pretrained=True,backbone = 'ResNet34', n_classes=1,module = 'MSMGM')#(n_classes=1,n_channels = 3).to(device)
    elif args['model'] == 'Morpho_UNet_ResNet34_MSMCM':
        model = Morpho_UNet_ResNet34(pretrained=True,backbone = 'ResNet34', n_classes=1,module = 'MSMCM')#(n_classes=1,n_channels = 3).to(device)
    elif args['model'] == 'Morpho_UNet_ResNet34_MSMOM':
        model = Morpho_UNet_ResNet34(pretrained=True,backbone = 'ResNet34', n_classes=1,module = 'MSMOM')#(n_classes=1,n_channels = 3).to(device)
    elif args['model'] == 'Morpho_UNet_efficientnetb4_MSMGM':
        model = Morpho_UNet_efficientnetb4(pretrained=True,backbone = 'efficientnet_b4', n_classes=1)#(n_classes=1,n_channels = 3).to(device)
    elif args['model'] == 'Morpho_UNet_efficientnetb4_MSMCM':
        model = Morpho_UNet_efficientnetb4(pretrained=True,backbone = 'efficientnet_b4', n_classes=1)#(n_classes=1,n_channels = 3).to(device)
    elif args['model'] == 'Morpho_UNet_efficientnetb4_MSMOM':
        model = Morpho_UNet_efficientnetb4(pretrained=True,backbone = 'efficientnet_b4', n_classes=1)#(n_classes=1,n_channels = 3).to(device)
        
        
   # total parameters and trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # loss function
    if args['loss'] == 'mse':
        criterion = nn.MSELoss()
    elif args['loss'] == 'diceloss':
        criterion = smp.losses.DiceLoss(mode= 'binary', classes=None, log_loss=False, from_logits=True, smooth=0.0, eps=1e-07)
    elif args['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    elif args['loss'] == 'BCEDiceLoss':
        criterion = BCEDiceLoss().to(device)
    # initialize SaveBestModel class
    save_best_model = SaveBestModel()
    save_best_model1 = SaveBestModel1()

    # lists to keep track of losses and accuracies
    train_loss, valid_loss,test_loss = [], [],[]
    train_acc, valid_acc,test_acc = [], [], []

    # start the training
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch+1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,  
                                                    criterion)
        test_epoch_loss, test_epoch_acc = validate(model, test_loader,  
                                                    criterion)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        test_loss.append(test_epoch_loss)

        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        test_acc.append(test_epoch_acc)

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print(f"Test loss: {test_epoch_loss:.3f}, test acc: {test_epoch_acc:.3f}")
        # save the best model till now if we have the least loss in the current epoch
        save_best_model(
            valid_epoch_loss, epoch, model, optimizer, criterion,savePath)
        save_best_model1(
            valid_epoch_acc, epoch, model, optimizer, criterion,savePath)
        print('-'*50)
        # scheduler.step()

    # save the trained model weights for a final time
    save_model(epochs, model, optimizer, criterion,savePath)
    # save the loss and accuracy plots
    save_plots(train_acc, valid_acc, test_acc,train_loss, valid_loss, test_loss,  savePath)
    print('TRAINING COMPLETE')

