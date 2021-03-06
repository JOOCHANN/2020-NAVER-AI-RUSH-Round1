import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from argparse import ArgumentParser
from pathlib import Path
from nsml.constants import DATASET_PATH
import nsml
import torch
import os
import pandas as pd
import numpy as np

from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

from warmup_scheduler import GradualWarmupScheduler
from dataset import SPAM
from utils import *

classes = ['normal', 'monotone', 'screenshot', 'unknown']

seed = '1'
lr = 0.01
max_epoch = 7
print_freq = 50
model_name = 'efficientnet-b7'
num_classes = len(classes)
weight_decay = 5e-04
momentum = 0.9
batch = 16
stepsize = 1
is_scheduler = True
gamma = 0.1

def bind_model(model):
    def save(dirname, *args):
        checkpoint = {'model': model.state_dict()}
        torch.save(checkpoint, os.path.join(dirname, 'model.pt'))
        print("saved")

    def load(dirname, *args):
        checkpoint = torch.load(os.path.join(dirname, 'model.pt'))
        model.load_state_dict(checkpoint['model'])
        print("loaded")

    def infer(data, **kwargs):
        spam_test = SPAM(Path(data), isTrain = False)
        testloader = DataLoader(dataset=spam_test, batch_size=batch, shuffle=False)
        model.eval()
        predictions = []
        filenames = []
        with torch.no_grad():
            for i, (data, f) in enumerate(testloader):
                data = data.cuda()
                outputs = model(data)
                o = outputs.data.max(1)[1].cpu().tolist()
                predictions = predictions + o
                filenames = filenames + list(f)

        y_pred = np.array(predictions)
        print('len filenames', len(filenames))
        print('len y_pred', len(y_pred))
        ret = pd.DataFrame({'filename': filenames, 'y_pred': y_pred})
        print('Done')
        return ret

    nsml.bind(save=save, load=load, infer=infer)

def main(pause):
    torch.manual_seed(seed)
    use_gpu = torch.cuda.is_available()
    print("Use gpu :", use_gpu)
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)

    if pause == 0:
        print("Creating dataset loading")
        spam_trainset = SPAM(Path(DATASET_PATH), isTrain = True)
        trainloader = DataLoader(dataset=spam_trainset, batch_size=batch, shuffle=True)
        print("End dataset loading")


    print("Creating model: {}".format(model_name))
    model = EfficientNet.from_pretrained(model_name, num_classes=num_classes)
    print('Model input size', EfficientNet.get_image_size(model_name))
    print("End Creating model: {}".format(model_name))

    # if use_gpu:
    #     model = nn.DataParallel(model).cuda()
    print("Loading model to GPU")
    model.to("cuda")
    # model = nn.DataParallel(model).cuda()
    print("Loaded")

    #Loss Function
    criterion = nn.CrossEntropyLoss()
    #Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)

    #learning rate??? ?????? epoch?????? ??????
    if is_scheduler == True:
        # ??? stepsize?????? learning rate??? 0.1??? ???????????? scheduler ??????
        scheduler = lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

        # CyclicLR
        # scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=1e-2, step_size_up=2000)

        # warmup
        # scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, 
        # total_epoch=1, after_scheduler=scheduler)


    #nsml model bind
    bind_model(model)

    if pause != 0:
        nsml.paused(scope=locals())

    for epoch in range(max_epoch):
        epoch = epoch
        print("==> Epoch {}/{}".format(epoch+1, max_epoch))

        #model train
        train(model, criterion, optimizer, trainloader, use_gpu, epoch)

        # ??? stepsize?????? learning rate??? 0.1??? ???????????? scheduler ??????.
        if is_scheduler == True:
            if epoch == 0:
                print("epoch 0... no learning rate step")
            else : 
                scheduler.step()

        #Train Accuracy??? ??????
        print("==> Train")
        acc, err = test(model, trainloader, use_gpu)
        print("Accuracy (%): {}\t Error rate(%): {}".format(acc, err))

        #save model
        nsml.save(checkpoint=epoch)
        # save_model(model, 'e{}'.format(epoch))

def train(model, criterion, optimizer, trainloader, use_gpu, epoch):
    model.train()
    losses = AverageMeter() # loss??? ????????? ????????? ??????
    for i, (data, labels) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()

        outputs = model(data) # ??? ?????????????????? softmax?????? output?????? ??????, ????????? batchsize??????
        loss = criterion(outputs, labels) # ?????? label??? output??? ???????????? loss ??????
        optimizer.zero_grad() # optimizer ?????????
        loss.backward() # loss backpropagation
        optimizer.step() # parameter update
        # AverageMeter function??? update?????? ?????? -> loss??? ????????? losses??? ??????
        losses.update(loss.item(), labels.size(0)) # labels.size(0) = batch size
    
        if (i+1) % print_freq == 0: #??? print_freq iteration?????? ????????? ??????
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})\t lr {}" \
                    .format(i+1, len(trainloader), losses.val, losses.avg, lr))

        if (i+1) % 540 == 0 :
            name = str(epoch) + '_' + str(i+1)
            nsml.save(checkpoint=name)
            print('save :', name)
            # for param_group in optimizer.param_groups:
            #     if epoch > 0:
            #         param_group['lr'] = param_group['lr'] * 0.5

            #print("change learning rate :", param_group['lr'])

def test(model, testloader, use_gpu):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad(): #parameter??? ???????????? ??????(backpropagation??? ????????????)
        for i, (data, labels) in enumerate(testloader):
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs = model(data)
            # Softmax??? ?????????????????? outputs??? ???????????? ????????? ??? ????????? ?????? ?????? ????????? index??? ????????????
            predictions = outputs.data.max(1)[1]
            # labels.size(0) = batch size????????? total??? ?????? ??? ???????????? ??? ??????
            total += labels.size(0)
            # ????????? predictions??? index?????? labels??? ?????? ????????? correct??? count???
            #predictions??? labels??? ??????????????? .sum()??? ???????????? count??????
            correct += (predictions == labels.data).sum()

            if i == 100:
                break

    # acc = ????????? ?????? ?????? / ?????? ?????????
    acc = correct.item() * 1.0 / total
    # err = 1 - acc
    err = 1.0 - acc
    return acc, err

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--mode', default='train')
    parser.add_argument('--pause', default=0)
    args = parser.parse_args()

    if args.pause:
        print("Testing!!!")
        main(pause = args.pause)
    
    if args.mode == 'train':
        print("Training!!!")
        main(pause = args.pause)