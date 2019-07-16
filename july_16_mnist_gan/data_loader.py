import torch
from torch.utils.data.dataset import Dataset

import torchvision
import torchvision.transforms as transforms

import os

from PIL import Image

def len_sum(list,range_number):
    result=0
    for i in range(range_number):
        result+=len(list[i])
    return result

class read_dataset(Dataset):
    def __init__(self,data_dir):
        self.data_list=[[] for _ in range(10)]
        for i in range(10):
            self.data_list[i]=[data_dir+str(i)+'/'+file_name for file_name in os.listdir(data_dir+str(i))]

        self.transform=transforms.Compose([transforms.ToTensor()])
    def __len__(self):
        return len_sum(self.data_list,10)
    def __getitem__(self,index):
        if index<len_sum(self.data_list,1):
            img=Image.open(self.data_list[0][index])
        elif index<len_sum(self.data_list,2):
            img=Image.open(self.data_list[1][index-len_sum(self.data_list,1)])
        elif index<len_sum(self.data_list,3):
            img=Image.open(self.data_list[2][index-len_sum(self.data_list,2)])
        elif index<len_sum(self.data_list,4):
            img=Image.open(self.data_list[3][index-len_sum(self.data_list,3)])
        elif index<len_sum(self.data_list,5):
            img=Image.open(self.data_list[4][index-len_sum(self.data_list,4)])
        elif index<len_sum(self.data_list,6):
            img=Image.open(self.data_list[5][index-len_sum(self.data_list,5)])
        elif index<len_sum(self.data_list,7):
            img=Image.open(self.data_list[6][index-len_sum(self.data_list,6)])
        elif index<len_sum(self.data_list,8):
            img=Image.open(self.data_list[7][index-len_sum(self.data_list,7)])
        elif index<len_sum(self.data_list,9):
            img=Image.open(self.data_list[8][index-len_sum(self.data_list,8)])
        else:
            img=Image.open(self.data_list[9][index-len_sum(self.data_list,9)])
        img=self.transform(img)
        label=torch.ones(1)
        return img,label