import clip

try:
    import os
    import torch
    import torch.nn as nn
    import torchvision
    import numpy as np
    import pickle
    # from utils.resnet import resnet18, wide_resnet50_2
    from torchvision.models import resnet18, wide_resnet50_2
    import torchvision.transforms as transforms
    import warnings
    import torchvision.datasets as datasets
    from clip.model import *
    import traceback

    warnings.filterwarnings("ignore")
except Exception as e:
    print(e)
    raise e


def resnet(num_classes, layer=50, pretrain=False):
    vision_heads = 32
    vision_layers = (3, 4, 6, 3)
    embed_dim = 1024
    vision_heads = 32
    image_resolution = 224
    vision_width = 64

    visual_encoder = ModifiedResNet(
        layers=vision_layers,
        output_dim=embed_dim,
        heads=vision_heads,
        input_resolution=image_resolution,
        width=vision_width,
        num_classes=num_classes
    )

    if pretrain:
        temp, _ = clip.load('RN50', 'cpu')
        pretrained = temp.visual.state_dict()

        visual_encoder_dict = visual_encoder.state_dict()
        state_dict = {k: v for k, v in pretrained.items() if k in visual_encoder_dict.keys()}
        visual_encoder_dict.update(state_dict)
        visual_encoder.load_state_dict(visual_encoder_dict)
    return visual_encoder


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))  # args.lr = 0.1 , 即每30步，lr = lr /10
    for param_group in optimizer.param_groups:  # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
        param_group['lr'] = lr


def train(dataset, batch_size, num_epoch, pretrain, fc, device_ids, lr=0.1):
    ckp_path = os.path.join('./checkpoint', dataset)
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    traindir = os.path.join('/userhome/datasets/pets', 'train')
    valdir = os.path.join('/userhome/datasets/pets', 'test')
    # traindir = os.path.join('D:\datasets\pets', 'train')
    # valdir = os.path.join('D:\datasets\pets', 'test')
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))


    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            # transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    num_classes = len(train_dataset.classes)


    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=16, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

    model = resnet(num_classes=num_classes, pretrain=pretrain)

    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device_ids)

    if fc:
        optimizer = torch.optim.SGD(model.fc.parameters(), lr, momentum=0.9, weight_decay=1e-4)
        optimizer = torch.optim.Adam(model.fc.parameters(), 3e-4)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-4)
        optimizer = torch.optim.Adam(model.parameters(), 3e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 90], gamma=0.1)
    loss_func = torch.nn.CrossEntropyLoss().to(device_ids)

    it = 0
    best_acc = 0

    for epoch in range(num_epoch):
        losses = 0
        corrects = 0
        total = 0

        # adjust_learning_rate(lr, optimizer, epoch)

        # train
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device_ids), labels.to(device_ids)
            #num_labels = round(batch_size*scale)
            #label_noise = (torch.rand(num_labels)*1000).long()
            #labels[0:num_labels] = label_noise
            prob = model(images)
            loss = loss_func(prob, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += (loss * len(images)).item()
            pred_idx = torch.argmax(prob.detach(), 1)
            corrects += (pred_idx == labels).sum().item()

            total += len(images)

        print('train epoch:{}\tloss:{:.4f}\tacc:{:.4f}'.format(epoch, losses / total,
                                                               corrects / total))

        # eval
        model.eval()
        with torch.no_grad():

            corrects = 0
            total = 0

            for images, labels in val_loader:
                images, labels = images.to(device_ids), labels.to(device_ids)
                prediction = model(images)
                pred_idx = torch.argmax(prediction.detach(), 1)
                corrects += (pred_idx == labels).sum().item()
                total += len(images)
            print('acc:{:.4f}'
                  .format(corrects / total))

            if corrects / total > best_acc:
                best_acc = corrects / total
                print(best_acc)
                torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},
                           os.path.join(ckp_path,
                                        'resnet50_fc.pt'))
        scheduler.step()


import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc', type=bool, default=False)
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.1)
    opt = parser.parse_args()
    try:
        device = 'cuda:0'
        # device = 'cpu'
        train(dataset='pets', fc=opt.fc, pretrain=opt.pretrain, num_epoch=120, batch_size=128, lr=opt.lr, device_ids=device)
    except Exception as e:
        print(traceback.format_exc())
        raise e



