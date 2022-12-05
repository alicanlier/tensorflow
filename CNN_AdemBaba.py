# -- coding: utf-8 --

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.io import read_image
from tqdm.notebook import tqdm, trange
import glob
import os
from os import listdir
from os.path import isfile, join
import pylab as py
from torchvision.utils import save_image


Nparam = 32
Npix = 256
class CNN(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(1, Nparam, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(Nparam, Nparam, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(Nparam, Nparam*2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(Nparam*2, Nparam*2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(Nparam*2*(Npix//2//2)**2, 256)
        self.fc2 = nn.Linear(256, Npix**2)

    def forward(self, x):
        
        #conv layer 1 no pooling
        x = self.conv1(x)
        x = F.relu(x)
        
        # conv layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # conv layer 3 no pooling
        x = self.conv3(x)
        x = F.relu(x)
        
        #conv layer 4 
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        # fc layer 1
        x = x.view(-1, Nparam*2*(Npix//2//2)**2)
        x = self.fc1(x)
        x = F.relu(x)
        
        # fc layer 2
        x = self.fc2(x)
        return x        

result_dir = r'D:\\onedrive2\\OneDrive\\research2\\projects\\dida_machine_learning_project\\results\\'
image_dir = r'D:\\onedrive2\\OneDrive\\research2\\projects\\dida_machine_learning_project\\dida_test_task\\images\\'
label_dir =  r'D:\\onedrive2\\OneDrive\\research2\\projects\\dida_machine_learning_project\\dida_test_task\\labels\\'
label_flnms = [f for f in listdir(label_dir) if isfile(join(label_dir, f))]
#label_full_flnms = [join(label_dir,f) for f in label_flnms]
#image_training_flnms = [join(image_dir,f) for f in label_flnms]
image_test_flnms = [f for f in listdir(image_dir) if isfile(join(image_dir, f)) and f not in label_flnms]


if  not all([isfile(join(image_dir,f)) for f in label_flnms]):
    print("One or more image file(s) is(are) missing.")
    assert 0


training_data_set = torch.FloatTensor(len(label_flnms),1,256,256)
training_label_set = torch.FloatTensor(len(label_flnms),1,256,256)
for i, flnm in enumerate(label_flnms):
    training_data_set[i] = read_image(join(image_dir,flnm))[:1]/256.
    training_label_set[i] = read_image(join(label_dir,flnm))[:1]/256.

test_data_set = torch.FloatTensor(len(image_test_flnms),1,256,256)
for i, flnm in enumerate(image_test_flnms):
    test_data_set[i] = read_image(join(image_dir,flnm))[:1]/256.
        
        
model = CNN()
# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  
optimizer =torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#optimizer =torch.optim.SparseAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
# Iterate through train set minibatchs 
for epoch in trange(1000):
    # Zero out the gradients
    optimizer.zero_grad()
    
    x = training_data_set
    y = model(x)
    loss = criterion(y, training_label_set.view(-1,256*256))
    print(epoch, loss)

    # Backward pass
    loss.backward()
    optimizer.step()
    
    if epoch % 5 ==0:
        py.imshow(model(test_data_set[0]).detach().numpy().reshape(256,256,1), cmap='gray')
        py.show()

        
        
for i,flnm  in enumerate(image_test_flnms):
    save_image(model(test_data_set[i]).view(1,256,256), join(result_dir, flnm))
    
# prediction = model(test_data_set)