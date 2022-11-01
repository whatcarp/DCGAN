#-------------------------------------#
#   生成1x1的图片和5x5的图片
#-------------------------------------#
import os
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from dcgan_nets import generator


class DCGAN:
    _defaults = {
        "channel": 128,
        "input_shape": [64, 64],
        "weights_path": '',
        "device": torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value
        self.create_G_model()

    # 创建G_model
    def create_G_model(self):
        self.net = generator(self.channel, self.input_shape)
        self.net.load_state_dict(torch.load(self.weights_path, map_location=self.device))  # 加载权重
        print(f'{self.weights_path} model weights loaded.')
        self.net = self.net.to(self.device).eval()

    # 生成5×5的图片
    def generate_5x5_image(self, save_path):
        with torch.no_grad():  # 停止autograd模块的工作，以加速节省显存

            randn_in = torch.randn((5*5, 100)).to(self.device)  # 生成25个[100,]噪声
            G_imgs = self.net(randn_in)  # 送入net得到生成图像

            fig, ax = plt.subplots(5, 5, figsize=(5, 5))
            for i, j in itertools.product(range(5), range(5)):  # 双重迭代方法1
                ax[i, j].get_xaxis().set_visible(False)
                ax[i, j].get_yaxis().set_visible(False)

            for k in range(5*5):  # 双重迭代方法2
                i, j = k // 5, k % 5
                G_img = G_imgs[k].cpu().data.numpy().transpose(1, 2, 0)
                G_img = np.uint8((G_img * 0.5 + 0.5) * 255)  # 反归一化
                ax[i, j].cla()
                ax[i, j].imshow(G_img)

            label = 'Predict_5x5_results'
            fig.text(0.5, 0.04, label, ha='center')
            plt.savefig(save_path)

    # 生成1×1的图片
    def generate_1x1_image(self, save_path):
        with torch.no_grad():
            randn_in = torch.randn((1, 100)).to(self.device)
            G_imgs = self.net(randn_in)
            G_img = G_imgs[0].cpu().data.numpy().transpose(1, 2, 0)
            G_img = np.uint8((G_img * 0.5 + 0.5) * 255)  # 反归一化
            Image.fromarray(np.uint8(G_img)).save(save_path)


if __name__ == "__main__":

    os.mkdir("results") if not os.path.exists("results") else None
    save_path_5x5 = "results/predict_5x5_results.png"
    save_path_1x1 = "results/predict_1x1_results.png"

    dcgan = DCGAN(weights_path='pth_dir\G_Epoch10-GLoss2.006-DLoss0.4108.pth')
    while True:
        img = input('Just Press Enter~')
        dcgan.generate_5x5_image(save_path_5x5)
        dcgan.generate_1x1_image(save_path_1x1)
