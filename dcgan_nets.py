import math
import torch
import torch.nn as nn


def conv_out(size, stride) -> int:
    return int(math.ceil(float(size) / float(stride)))


class generator(nn.Module):
    def __init__(self, d=128, input_shape=[64, 64]):
        super().__init__()
        s_h, s_w = input_shape[0], input_shape[1]
        s_h2, s_w2 = conv_out(s_h, 2), conv_out(s_w, 2)
        s_h4, s_w4 = conv_out(s_h2, 2), conv_out(s_w2, 2)
        s_h8, s_w8 = conv_out(s_h4, 2), conv_out(s_w4, 2)
        self.s_h16, self.s_w16 = conv_out(s_h8, 2), conv_out(s_w8, 2)

        self.linear = nn.Linear(100, self.s_h16 * self.s_w16 * d * 8)  # d*8 × s_h16 × s_w16
        self.linear_bn = nn.BatchNorm2d(d * 8)  # BN:对一个batch中的所有样本的同一个channel的数据元素进行标准化处理，参数为通道数

        # ConvTranspose2d's para:[input_channels] [output_channels] [kernelsize] [stride]
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)

        self.deconv2 = nn.ConvTranspose2d(d * 4, d * 2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d * 2)

        self.deconv3 = nn.ConvTranspose2d(d * 2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)

        self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)

        self.relu = nn.ReLU()

        self.weight_init()

    def weight_init(self):  # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        bs, _ = x.size()  # 获取batch_size
        x = self.linear(x)
        x = x.view([bs, -1, self.s_h16, self.s_w16])  # size([bs, (d*8*s_w16*s_h16)]) -> size([bs, d*8, s_w16, s_h16])
        x = self.relu(self.linear_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        x = torch.tanh(self.deconv4(x))  # 归一化一手
        return x


class discriminator(nn.Module):
    def __init__(self, d=128, input_shape=[64, 64]):
        super().__init__()
        s_h, s_w = input_shape[0], input_shape[1]
        s_h2, s_w2 = conv_out(s_h, 2), conv_out(s_w, 2)
        s_h4, s_w4 = conv_out(s_h2, 2), conv_out(s_w2, 2)
        s_h8, s_w8 = conv_out(s_h4, 2), conv_out(s_w4, 2)
        self.s_h16, self.s_w16 = conv_out(s_h8, 2), conv_out(s_w8, 2)

        # Conv2d(输入通道数，输出通道数（卷积核个数），卷积核尺寸，步长，padding)
        # 输出尺寸 = （输入尺寸inputsize - kernel_size + 2 × padding）/ stride + 1
        # 例如：输入尺寸大小 256*256，kernel_size4*4，strides=2，padding=1。根据公式计算得到128*128

        # 64,64,3 -> 32,32,128
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)

        # 32,32,128 -> 16,16,256
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)

        # 16,16,256 -> 8,8,512
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)

        # 8,8,512 -> 4,4,1024
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)

        # 4,4,1024 -> 1
        self.linear = nn.Linear(self.s_h16 * self.s_w16 * d * 8, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.weight_init()

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0.1, 0.02)
                m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        bs, _, _, _ = x.size()  # 获取batch_size
        x = self.leaky_relu(self.conv1(x))
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = self.leaky_relu(self.conv4_bn(self.conv4(x)))
        x = x.view([bs, -1])
        x = self.linear(x)

        return x.squeeze()  # 删除大小为1的维度 如 [[1]] -> [1]


if __name__ == "__main__":
    from torchsummary import summary
    # 需要使用device来指定网络在GPU还是CPU运行
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    G_model = generator().to(device)
    summary(G_model, input_size=(100,))

    D_model = discriminator().to(device)
    summary(D_model, input_size=(3, 64, 64))

    # for i in G_model.state_dict():
    #     print(i,'\t',G_model.state_dict()[i].size())
