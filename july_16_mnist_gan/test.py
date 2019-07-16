import torch

from matplotlib import pyplot as plt

import model

if __name__=='__main__':
    net=model.Generator()

    # 네트워크 데이터 로드

    load_net='./g_net_epoch_10.pth'
    net.load_state_dict(torch.load(load_net))

    with torch.no_grad():
        z=torch.randn(1,64)
        fake_image=net(z)

        plt.imshow(fake_image.view(28,28),cmap='gray')
        plt.show()