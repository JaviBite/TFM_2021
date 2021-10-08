import numpy as np
from skimage.transform import resize
from skimage.io import imread
import cv2
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms  
import torch.optim as optim
import os

def capture(filename,timesep,rgb,h,w):
    tmp = []
    frames = np.zeros((timesep,rgb,h,w), dtype=np.float)
    i=0
    vc = cv2.VideoCapture(filename)
    if vc.isOpened():
        rval , frame = vc.read()
    else:
        rval = False
    frm = resize(frame,(h, w,rgb))
    frm = np.expand_dims(frm,axis=0)
    frm = np.moveaxis(frm, -1, 1)
    if(np.max(frm)>1):
        frm = frm/255.0
    frames[i][:] = frm
    i +=1
    while i < timesep:
        tmp[:] = frm[:]
        rval, frame = vc.read()
        frm = resize(frame,( h, w,rgb))
        frm = np.expand_dims(frm,axis=0)
        if(np.max(frm)>1):
            frm = frm/255.0
        frm = np.moveaxis(frm, -1, 1)
        frames[i-1][:] = frm # - tmp
        i +=1

    return frames


class TimeWarp(nn.Module):
    def __init__(self, baseModel, method='sqeeze'):
        super(TimeWarp, self).__init__()
        self.baseModel = baseModel
        self.method = method
 
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        if self.method == 'loop':
            output = []
            for i in range(time_steps):
                #input one frame at a time into the basemodel
                x_t = self.baseModel(x[:, i, :, :, :])
                # Flatten the output
                x_t = x_t.view(x_t.size(0), -1)
                output.append(x_t)
            #end loop
            #make output as  ( samples, timesteps, output_size)
            x = torch.stack(output, dim=0).transpose_(0, 1)
            output = None # clear var to reduce data  in memory
            x_t = None  # clear var to reduce data  in memory
        else:
            # reshape input  to be (batch_size * timesteps, input_size)
            x = x.contiguous().view(batch_size * time_steps, C, H, W)
            x = self.baseModel(x)
            x = x.view(x.size(0), -1)
            #make output as  ( samples, timesteps, output_size)
            x = x.contiguous().view(batch_size , time_steps , x.size(-1))
        return x

class extractlastcell(nn.Module):
    def forward(self,x):
        out , _ = x
        return out[:, -1, :]

def create_model(device, num_features = 12800, weights = None):
    # Create model
    num_classes = 1
    dr_rate= 0.2
    rnn_hidden_size = 80
    rnn_num_layers = 1

    if weights is not None:
        pretrained = True
    
    baseModel = models.vgg19(pretrained=pretrained).features

    i = 0
    for child in baseModel.children():
        if i < 28:
            for param in child.parameters():
                param.requires_grad = False
        else:
            for param in child.parameters():
                param.requires_grad = True
        i +=1

    num_features = 12800
    # Example of using Sequential
    model = nn.Sequential(TimeWarp(baseModel,method='loop'),
                        nn.LSTM(num_features, rnn_hidden_size, rnn_num_layers , batch_first=True , bidirectional=True ),
                        extractlastcell(),
                            nn.Linear(160, 256),
                        nn.ReLU(),
                        nn.Dropout(dr_rate),
    nn.Linear(256, num_classes))
    if weights is not None:
        checkpoint = torch.load(wight)
        model.load_state_dict(checkpoint['state_dict'])

    return model

def train(net, trainloader, path="./checkpoint.pth"):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()

    for epoch in range(2):  # loop over the dataset multiple times
    
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    torch.save(net.state_dict(), path)

print('Finished Training')

def pred(model,video,acuracy=0.58):
    model.eval()
    outputs = model(video)
    torch.sigmoid(outputs)
    preds = torch.sigmoid(outputs).to('cpu')
    preds = preds.detach().numpy()
    if preds[0][0] >=acuracy:
        return True , preds[0][0]
    else:
        return False , preds[0][0]