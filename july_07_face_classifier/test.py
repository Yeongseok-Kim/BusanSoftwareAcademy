import torch
import torch.nn as nn

import torchvision
import torchvision.transforms as transforms

from PIL import Image

import train

# 디바이스 식별

device='cuda' if torch.cuda.is_available() else 'cpu'

# 학습 데이터 로드

model=train.CNN().to(device)

load_net='./model_epoch_80.pth'
model.load_state_dict(torch.load(load_net))

# 테스트

transform=transforms.Compose([transforms.Resize((64,64)),transforms.Grayscale(1),transforms.ToTensor()])

with torch.no_grad():
    test_img=Image.open('./test.jpg')
    test_img=transform(test_img).to(device)
    test_result=model(test_img.unsqueeze(0))
    print(test_result)