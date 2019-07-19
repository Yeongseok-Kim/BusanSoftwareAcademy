import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets

from PIL import Image

import os

# 디바이스 식별

device='cuda' if torch.cuda.is_available() else 'cpu'

# 데이터셋 클래스

class CustomDataset(Dataset):
    def __init__(self,face_data_dir,nonface_data_dir):

        # face_data_dir 아래의 모든 파일 목록인 face_data_list

        self.face_data_list=[face_data_dir+file_name for file_name in os.listdir(face_data_dir)]
        
        # nonface_data_dir 아래의 모든 파일 목록인 nonface_data_list
        
        self.nonface_data_list=[nonface_data_dir+file_name for file_name in os.listdir(nonface_data_dir)]
        self.transform=transforms.Compose([transforms.Resize((64,64)),transforms.Grayscale(1),transforms.ToTensor()])
    def __len__(self):

        # 데이터셋 길이 반환

        return len(self.face_data_list+self.nonface_data_list)
    def __getitem__(self,index):
        if index<len(self.face_data_list):
            img=Image.open(self.face_data_list[index])
            img=self.transform(img)
            label=torch.ones(1)
        else:
            img=Image.open(self.nonface_data_list[index-len(self.face_data_list)])
            img=self.transform(img)
            label=torch.zeros(1)
        
        # img,label 반환

        return img,label
        
# 트레이닝 데이터 로드

batch_size=64
train_data=CustomDataset('./face_data/train_data/face/','./face_data/train_data/nonface/')
train_set=DataLoader(train_data,batch_size=batch_size,shuffle=True)

# 테스트 데이터 로드

test_data=CustomDataset('./face_data/test_data/face/','./face_data/test_data/nonface/')
test_set=DataLoader(test_data,len(test_data),shuffle=True)

# 트레이닝

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(1,32,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.layer2=nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2))
        self.fc=nn.Linear(16*16*64,1)
        nn.init.xavier_uniform_(self.fc.weight)
    def forward(self,x):
        out=self.layer1(x)
        out=self.layer2(out)
        out=out.view(out.size(0),-1)
        out=self.fc(out)
        return out

learning_rate=0.01
training_epochs=80

model=CNN().to(device)
criterion=nn.MSELoss().to(device)
optimizer=optim.Adam(model.parameters(),learning_rate)

total_batch=len(train_set)
print('Learning started. It takes sometime.')

for epoch in range(training_epochs):
    avg_cost=0
    
    for X,Y in train_set:
        X=X.to(device)
        Y=Y.to(device)

        optimizer.zero_grad()
        hypothesis=model(X)
        cost=criterion(hypothesis,Y)
        cost.backward()
        optimizer.step()

        avg_cost=cost/total_batch
    
    print('[Epoch: {:>4}] cost={:>.9}'.format(epoch+1,avg_cost))

    # 학습 데이터 저장

    torch.save(model.state_dict(),'./model_epoch_%d.pth'%(epoch+1))

print('Learning Finished!')