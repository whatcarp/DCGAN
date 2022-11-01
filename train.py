import os
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import DCganDataset, gen_annotation
from dcgan_nets import generator, discriminator


# ==============参数设置================
EPOCHS = 30
BATCH_SIZE = 4
channel = 128  # 特征层最厚是多少channel
input_shape = [64, 64]
Init_lr = 2e-3
momentum = 0.5
G_model_path = ""  # G预训练权重路径
D_model_path = ""  # D预训练权重路径
generate_flag = True  # 是否在每轮epoch后生成图片
# =====================================


save_dir = 'pth_dir'  # 训练时权重的保存路径
cudnn.benchmark = True  # 卷积加速(输入尺寸固定时)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 生成GD模型
G_model = generator(channel, input_shape)
D_model = discriminator(channel, input_shape)

# 加载模型权重
"""
    当新定义的网络（model_dict）和预训练网络（pretrained_dict）的层名不严格相等时，
    需要先将pretrained_dict里不属于model_dict的键剔除掉，
    再用预训练模型参数更新model_dict，
    最后用load_state_dict方法初始化自己定义的新网络结构。
"""
if G_model_path != "" and os.path.exists(G_model_path): 
    model_dict = G_model.state_dict()
    pretrained_dict = torch.load(G_model_path, map_location=DEVICE)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    G_model.load_state_dict(model_dict)
    print('G model loaded.')

if D_model_path != "" and os.path.exists(D_model_path):
    model_dict = D_model.state_dict()
    pretrained_dict = torch.load(D_model_path, map_location=DEVICE)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    D_model.load_state_dict(model_dict)
    print('D model loaded.')


# train model prepare
G_model_train = G_model.to(DEVICE)
D_model_train = D_model.to(DEVICE)

# loss function
BCE_loss = nn.BCEWithLogitsLoss()

# optimizer
G_optim = optim.Adam(G_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999))
D_optim = optim.Adam(D_model_train.parameters(), lr=Init_lr, betas=(momentum, 0.999))

# lr schedule：ConsineAnnealing
G_lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=G_optim, T_max=EPOCHS, eta_min=Init_lr * 0.01)
D_lr_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=D_optim, T_max=EPOCHS, eta_min=Init_lr * 0.01)


# 生成训练集各图片路径
annotation_path = 'train_lines.txt'
if not os.path.exists(annotation_path):
    gen_annotation('./datasets/')
with open(annotation_path, encoding='utf-8') as f:
    lines = f.readlines()
    num_train = len(lines)

# 读取训练集，生成数据加载器
train_dataset = DCganDataset(lines, input_shape)
data_loader = DataLoader(train_dataset, shuffle=False, batch_size=BATCH_SIZE)


# GAN train logic
for epoch in range(EPOCHS):

    G_model.train()
    D_model.train()
    G_loss_sum, G_loss_avg = 0, 0
    D_loss_sum, D_loss_avg = 0, 0

    with tqdm(enumerate(data_loader), total=len(data_loader)) as loop:
        for step, data in loop:  # 每batch，注意这里不用(data,target)了因为我们自己写的data_loader只返回图片没返回标签

            # step: 第几个batch
            # data: 当前batch_size张图片

            # 准备工作
            batch_size = data.size()[0]  # 因为有可能最后一个step的数据不够一个batch_size，所以要这样取一下
            y_real = torch.ones(batch_size)  # 真标签
            y_fake = torch.zeros(batch_size)  # 假标签
            noise_1 = torch.randn((batch_size, 100))  # D_model训练用
            noise_2 = torch.randn((batch_size, 100))  # G_model训练用

            # 移至GPU
            with torch.no_grad():
                data = data.to(DEVICE)
                y_real = y_real.to(DEVICE)
                y_fake = y_fake.to(DEVICE)
                noise_1 = noise_1.to(DEVICE)
                noise_2 = noise_2.to(DEVICE)

            """-----先训练D_model，利用真假图片训练D_model，让D_model更准确-----"""

            D_optim.zero_grad()  # 先清空梯度！

            # 喂入真实数据
            D_result = D_model_train(data)  # 送入真实数据
            D_real_loss = BCE_loss(D_result, y_real)  # 计算损失
            D_real_loss.backward()  # 反向传播，计算梯度

            # 喂入虚假数据
            G_result = G_model_train(noise_1)  # 生成虚假数据
            D_result = D_model_train(G_result)  # 送入虚假数据
            D_fake_loss = BCE_loss(D_result, y_fake)  # 计算损失 （想要训练D_model,故标签为y_fake）
            D_fake_loss.backward()  # 反向传播，计算梯度（梯度累加）

            D_optim.step()  # 梯度更新

            """-----再训练G_model，尽可能地让D_model认为自己生成的图片是真实的-----"""

            G_optim.zero_grad()  # 梯度请空

            G_result = G_model_train(noise_2)  # 生成虚假数据
            D_result = D_model_train(G_result).squeeze()  # 送入虚假数据
            G_train_loss = BCE_loss(D_result, y_real)  # 计算损失 （想要训练G_model,故标签为y_true）
            G_train_loss.backward()  # 反向传播

            G_optim.step()  # 梯度更新

            # 尤其注意！ 训练Gmodel并反向传播时也会给Dmodel带来梯度，一定要再Dmodel训练前清空梯度！

            """-------------------------------------------------------------"""

            # 记录损失值
            G_loss_sum += G_train_loss.item()
            D_loss_sum += (D_real_loss.item() + D_fake_loss.item()) * 0.5

            G_loss_avg = G_loss_sum / (step+1)
            D_loss_avg = D_loss_sum / (step+1)

            loop.set_description(f'Epoch {epoch+1}/{EPOCHS}')
            loop.set_postfix(G_loss=G_loss_avg, D_loss=D_loss_avg, lr=G_optim.state_dict()['param_groups'][0]['lr'])

            if step > 10:
                break

    if True:  # 一轮epoch结束后

        # 学习率更新
        G_lr_schedule.step()
        D_lr_schedule.step()

        # 保存权重
        G_weights_path = os.path.join(save_dir, f'G_Epoch{epoch + 1}-GLoss{G_loss_avg:.4}-DLoss{D_loss_avg:.4}.pth')
        D_weights_path = os.path.join(save_dir, f'D_Epoch{epoch + 1}-GLoss{G_loss_avg:.4}-DLoss{D_loss_avg:.4}.pth')

        os.mkdir(save_dir) if not os.path.exists(save_dir) else None
        torch.save(G_model.state_dict(), G_weights_path)
        torch.save(D_model.state_dict(), D_weights_path)

        # 生成图片
        if generate_flag:
            with torch.no_grad():  # 停止autograd模块的工作，以加速节省显存

                G_model.eval()
                randn_in = torch.randn((5*5, 100)).to(DEVICE)  # 生成25个[100,]噪声
                G_imgs = G_model(randn_in)  # 送入net得到生成图像

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
                label = f'Epoch {epoch} Predict_5x5_results'
                fig.text(0.5, 0.04, label, ha='center')

                os.mkdir("results") if not os.path.exists("results") else None
                save_path = f"results/Epoch {epoch} predict_5x5_results.png"
                plt.savefig(save_path)
