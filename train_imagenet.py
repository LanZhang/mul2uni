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


def resnet(layer=50):
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
        width=vision_width
    )
    return visual_encoder


def adjust_learning_rate(lr, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (0.1 ** (epoch // 30))  # args.lr = 0.1 , 即每30步，lr = lr /10
    for param_group in optimizer.param_groups:  # 将更新的lr 送入优化器 optimizer 中，进行下一次优化
        param_group['lr'] = lr


def train(dataset, batch_size, num_epoch, device_ids, model_idx,
          lr=0.1, img_size=224):
    ckp_path = os.path.join('./checkpoint', str(model_idx), dataset)
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    traindir = os.path.join('/userdata/imagenet.zip/imagenet', 'train')
    valdir = os.path.join('/userdata/imagenet.zip/imagenet', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # random label 10%
    # random_size = int(len(train_dataset)//10)
    # label_noise = (torch.rand(random_size) * 1000).long()
    # for i, j in enumerate(random.sample(list(range(len(train_dataset))), random_size)):
    #     train_dataset.targets[j] = label_noise[i]

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

    model = resnet(50)

    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=0.9,
                                weight_decay=1e-4)
    loss_func = torch.nn.CrossEntropyLoss().cuda(device=device_ids[0])

    it = 0
    best_acc = 0

    for epoch in range(num_epoch):
        losses = 0
        corrects = 0
        total = 0

        adjust_learning_rate(lr, optimizer, epoch)

        # train
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
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
                images, labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
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
                                        'resnet50.pt'))


if __name__ == '__main__':
    try:
        device = [0]
        batch_size = 256
        train(dataset='ILSVRC2012', num_epoch=90, batch_size=batch_size, lr=0.1,
              model_idx=0, device_ids=device)
    except Exception as e:
        print(traceback.format_exc())
        raise e



