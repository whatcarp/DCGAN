import os
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset


class DCganDataset(Dataset):
    def __init__(self, annotation_lines, input_shape):
        super().__init__()

        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape

    # 必须重写
    def __len__(self):
        return self.length

    # 必须重写
    def __getitem__(self, index):  # 根据索引index，读取一个你返回的数据，遍历用

        # 读取图片
        image = Image.open(self.annotation_lines[index].split()[0])

        # 确定（转化）为RGB，并resize， ！注意宽高的顺序
        image = self.cvtColor(image).resize([self.input_shape[1], self.input_shape[0]], Image.BICUBIC)

        # 转化为numpy格式
        image = np.array(image, dtype=np.float32)

        # 归一化
        image = self.preprocess_input(image)

        # 将将通道数移到前面，已经变成torch.tensor的形状了
        image = np.transpose(image, (2, 0, 1))

        return image

    @staticmethod
    def cvtColor(image):
        # 确保三通道RGB图
        if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
            return image
        # 否则转化为RGB图
        else:
            image = image.convert('RGB')
            return image

    @staticmethod
    def preprocess_input(x):
        x /= 255
        x -= 0.5
        x /= 0.5
        return x


# def DCgan_dataset_collate(batch):
#     images = [image for image in batch]
#     images = torch.from_numpy(np.array(images, np.float32))
#     return images

def gen_annotation(datasets_path):
    photos_names = os.listdir(datasets_path)
    photos_names = sorted(photos_names)
    with open('train_lines.txt', 'w', encoding='utf-8') as list_file:
        for photo_name in photos_names:
            if (photo_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff'))):
                list_file.write(os.path.join(os.path.abspath(datasets_path), photo_name))
                list_file.write('\n')
