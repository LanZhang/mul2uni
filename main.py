import os
import torchvision.models
import numpy as np
import clip.clip
import torch
from torchvision.datasets import CIFAR100
from PIL import Image
from attack.imageAttack import *
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import csv

model_lists = clip.available_models()
device = "cuda" if torch.cuda.is_available() else "cpu"


class CommonDataset(Dataset):
    def __init__(self, root, data_dir, data_transform):
        self.data = []
        self.labels = []
        self.categories = []
        self.data_dir = data_dir
        self.root = root
        self.data_transform = transforms.Compose(data_transform)

        with open('D:\data\imagenet\categories.csv') as f_cate:
            cate_list = []
            cate_csv = csv.reader(f_cate)
            headers = next(f_cate)
            for row in cate_csv:
                cate_list.append(row[1])

        with open(self.data_dir) as f:
            f_csv = csv.reader(f)
            headers = next(f_csv)

            for row in f_csv:
                dir = row[0]
                label = int(row[6]) - 1
                self.data.append(os.path.join(self.root, dir + '.png'))
                self.labels.append(label)
                self.categories.append(cate_list[label])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        path = self.data[item]
        image = Image.open(path).convert('RGB')

        image = self.data_transform(image)
        label = self.labels[item]
        cate = self.categories[item]
        return image, label, cate

# features_in_hook = [None]
# features_out_hook = [None]
#
#
# def hook(module, fea_in, fea_out):
#     features_in_hook[0] = fea_in
#     features_out_hook[0] = fea_out
#     return None


def eval(batch_size=20):
    clip_model, preprocess = clip.load('RN50', device)
    su_model = clip_model.encode_image

    temp, _ = clip.load('RN50', 'cpu')
    temp = temp.visual
    a = temp.state_dict()
    # su_model = torchvision.models.resnet50(pretrained=True).to(device)
    images_preprocess = preprocess.transforms[0:4]
    images_normalize = preprocess.transforms[4]
    vic_model = torchvision.models.resnet101(pretrained=True).to(device)

    # a = list(vic_model.named_modules())
    # layer_name = 'fc'
    # for (name, module) in vic_model.named_modules():
    #     if name == layer_name:
    #         module.register_forward_hook(hook=hook)

    dataset = CommonDataset(root='D:\data\imagenet\images',
                            data_dir='D:\data\imagenet\images.csv',
                            data_transform=images_preprocess)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

    num_image = len(data_loader.dataset)

    image_attacker = ImageAttacker(8.0 / 255., preprocess=images_normalize, bounding=(0, 1), cls=False)

    with open('D:\data\imagenet\categories.csv') as f_cate:
        cate_list = []
        cate_csv = csv.reader(f_cate)
        headers = next(f_cate)
        for row in cate_csv:
            cate_list.append(row[1])


    category = clip.tokenize(cate_list).to(device)
    # 计算CLIP的准确率
    # correct = 0
    # with torch.no_grad():
    #     for images, labels, cate in data_loader:
    #         images, labels = images.to(device), labels.to(device)
    #         images = images_normalize(images)
    #
    #         logits_per_image, logits_per_text = clip_model(images, category)
    #         probs = logits_per_image.softmax(dim=-1)
    #         correct += (probs.argmax(1) == labels).sum().detach()
    #     print('corrects for CLIP:{:.4f}'.format(correct / num_image))


    # 计算CLIP的白盒对抗准确率
    correct_adv = 0
    for images, labels, cate in data_loader:
        images, labels = images.to(device), labels.to(device)
        images_adv = image_attacker.clip_ce(clip_model, 1, images, category, labels)
        images_adv = image_attacker.run_trades(net=su_model, num_iters=10, image=images)
        images_adv = images_normalize(images_adv)

        with torch.no_grad():
            # logits_adv = vic_model(images_adv)
            # correct_adv += (logits_adv.argmax(1) == labels).sum().detach()
            logits_per_image, logits_per_text = clip_model(images_adv, category)
            probs = logits_per_image.softmax(dim=-1)
            correct_adv += (probs.argmax(1) == labels).sum().detach()
    print('correct_adv:{:.4f}'.format(correct_adv/num_image))


    # 计算黑盒对抗准确率
    correct_adv = 0
    for images, labels, cate in data_loader:
        images, labels = images.to(device), labels.to(device)
        images_adv = image_attacker.run_trades(vic_model, images, 1)
        images_adv = images_normalize(images_adv)

        with torch.no_grad():

            logits_adv = vic_model(images_adv, inference=True)
            correct_adv += (logits_adv.argmax(1) == labels).sum().detach()

            # logits_per_image, logits_per_text = clip_model(images_adv, category)
            # probs = logits_per_image.softmax(dim=-1)
            # correct_adv += (probs.argmax(1) == labels).sum().detach()

    print('correct_adv:{:.4f}'.format(correct_adv/num_image))


eval()
