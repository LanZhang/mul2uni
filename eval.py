try:
    import clip
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
    import foolbox
    from foolbox.attacks import LinfPGD

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


def evaluation(dataset, batch_size, checkpoint, device_ids, eps=8):
    ckp_path = os.path.join('./checkpoint_pets', dataset)
    if not os.path.exists(ckp_path):
        os.makedirs(ckp_path)

    traindir = os.path.join('/userhome/datasets/pets', 'train')
    valdir = os.path.join('/userhome/datasets/pets', 'test')
    # traindir = os.path.join('D:\datasets\pets', 'train')
    # valdir = os.path.join('D:\datasets\pets', 'test')



    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # normalize,
        ]))

    num_classes = len(train_dataset.classes)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=16, pin_memory=True)

    model = resnet(num_classes=num_classes, pretrain=False)
    model.load_state_dict(torch.load(os.path.join(ckp_path, checkpoint)))

    # model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])

    attacker = LinfPGD(steps=10, rel_stepsize=1 / 8)
    preprocessing = dict(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), axis=-3)
    fnet = foolbox.PyTorchModel(model, (0, 1), device_ids[0], preprocessing)
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))


    it = 0
    best_acc = 0
    # eval
    model.eval()
    with torch.no_grad():
        corrects = 0
        total = 0

        for images, labels in val_loader:
            images, labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
            images = normalize(images)
            prediction = model(images)
            pred_idx = torch.argmax(prediction.detach(), 1)
            corrects += (pred_idx == labels).sum().item()
            total += len(images)
        print('checkpoint:{}\t clean acc:{:.4f}'
              .format(checkpoint, corrects / total))


    corrects = 0
    total = 0

    for images, labels in val_loader:
        images, labels = images.cuda(device=device_ids[0]), labels.cuda(device=device_ids[0])
        _, images_adv, _ = attacker(fnet, images, labels, epsilons=eps/255.0)
        images_adv = normalize(images_adv)
        prediction = model(images_adv)
        pred_idx = torch.argmax(prediction.detach(), 1)
        corrects += (pred_idx == labels).sum().item()
        total += len(images)
    print('checkpoint:{}\t adv acc:{:.4f}'
          .format(checkpoint, corrects / total))
