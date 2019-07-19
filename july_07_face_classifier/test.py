import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from PIL import Image

# 네트워크

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

model=CNN().to(device)

# 디바이스 식별

device='cuda' if torch.cuda.is_available() else 'cpu'

# 학습 데이터 로드

load_net='./model_epoch_80.pth'
model.load_state_dict(torch.load(load_net))

# 테스트

transform=transforms.Compose([transforms.Resize((64,64)),transforms.Grayscale(1),transforms.ToTensor()])

with torch.no_grad():
    test_img=Image.open('./test.jpg')
    test_img=transform(test_img).to(device)
    test_result=model(test_img.unsqueeze(0))
    print(test_result)