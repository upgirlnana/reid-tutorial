import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torchvision import datasets, transforms

from model import PCBModel
from dataset import *
from utils import *
from config import *

global best_mAP
best_mAP = 0.0

def parse_input():
  parser = argparse.ArgumentParser(description='Training arguments')
  parser.add_argument('--dataset', type=str, default='market1501', 
                      choices=['market1501', 'cuhk03', 'duke'])
  parser.add_argument('--resume', '-r', action='store_true',
                      help='resume from checkpoint')
  parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')  
  parser.add_argument('--batchsize', type=int, default=64, help='batch size')
  parser.add_argument('--stripes', type=int, default=6, help='number of stripes')
  arg = parser.parse_args()
  return arg


def train(model, trainloader, lr):
  criterion = nn.CrossEntropyLoss()

  # Finetune the net
  optimizer = optim.SGD([
    {'params': model.backbone.parameters(), 'lr': lr / 10},
    {'params': model.local_conv.parameters(), 'lr': lr},
    {'params': model.fc_list.parameters(), 'lr': lr}
  ], momentum=0.9, weight_decay=5e-4, nesterov=True)

  scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

  progress_bar = ProgressBar(total=len(trainloader))
  model.train()
  train_loss = 0
  correct = 0
  total = 0
  for batch_idx, (inputs, labels) in enumerate(trainloader):
    inputs, labels = Variable(make_cuda(inputs)), Variable(make_cuda(labels))
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = 0
    for feat in outputs:
      loss += criterion(feat, labels)
    loss.backward()
    optimizer.step()

    train_loss += loss.data.item()
    progress_bar.move(leftmsg="training", rightmsg="loss: %.2f" % (train_loss/(batch_idx+1)))


def test(model, dataloader, img_datasets):
  model.eval()

  gallery_cams, gallery_labels = get_cam_label(img_datasets['gallery'].imgs)
  query_cams, query_labels = get_cam_label(img_datasets['query'].imgs)

  current = 0
  batch_size = dataloader['gallery'].batch_size
  feature_num = len(dataloader['gallery'])+len(dataloader['query'])
  query_num = len(query_labels)
  gallery_num = len(gallery_labels)
  progress_bar = ProgressBar(total=feature_num+query_num)

  # extracting features

  gallery_features = torch.Tensor()
  query_features = torch.Tensor()

  for batch_idx, (img, _) in enumerate(dataloader['query']):
    feature = extract_feature(model, img)
    query_features = torch.cat((query_features, feature), dim=0)
    current += 1
    progress_bar.move(leftmsg="evaluate (%d/%d)" % (current, feature_num), rightmsg="")

  for batch_idx, (img, _) in enumerate(dataloader['gallery']):
    feature = extract_feature(model, img)
    current += 1
    progress_bar.move(leftmsg="evaluate (%d/%d)" % (current, feature_num), rightmsg="")
    gallery_features = torch.cat((gallery_features, feature), dim=0)

  fnorm = gallery_features.norm(p=2, dim=1, keepdim=True)
  gallery_features = gallery_features.div(fnorm)
  fnorm = query_features.norm(p=2, dim=1, keepdim=True)
  query_features = query_features.div(fnorm)

  # evaluate

  CMC, mAP = evaluate(query_features, query_labels, query_cams,
                      gallery_features, gallery_labels, gallery_cams, progress_bar)

  print('top1:%f top5:%f top10:%f mAP:%f' % (CMC[0], CMC[4], CMC[9], mAP))
  return mAP


if __name__ == '__main__':
  args = parse_input()
  config = Config()


  # 1. dataset

  print("\n==> Preparing dataset ... ", end='')
  raw_path = config.dataset_path[args.dataset]
  transformed_path = os.path.join(raw_path, 'transformed')
  assert os.path.isdir(raw_path), "raw path %s does not exist" % raw_path

  if not os.path.isdir(transformed_path):
    transform_dataset(raw_path, transformed_path)
    make_val(transformed_path)

  img_datasets, dataloader = prepare_dataset(transformed_path, args.batchsize)
  print("done!")

  # 2. load model
  start_epoch = 0

  checkpoint_dir_path = os.path.join('../checkpoint', args.dataset)
  checkpoint_name = 'btchsz' + str(args.batchsize) + '_' + 'strp' + str(args.stripes) + '.ckpt.t7'
  checkpoint_path = os.path.join(checkpoint_dir_path, checkpoint_name)
  make_dir(checkpoint_dir_path)

  if args.resume:  # restore from checkpoint
    print('\n==> Resuming model from checkpoint ... ', end='')
    assert os.path.isfile(checkpoint_path), 'checkpoint %s does not exist' % checkpoint_path
    checkpoint = torch.load(checkpoint_path)
    pcb = checkpoint['model']
    best_mAP = checkpoint['mAP']
    start_epoch = checkpoint['epoch']
    print("done! mAP:", best_mAP)
  else:
    print('\n==> Building model ... ', end='')
    pcb = PCBModel(num_stripes=args.stripes,
                   num_classes=len(img_datasets['train'].classes))
    print("done!")

  pcb = make_cuda(pcb)
  cudnn.benchmark = True

  # 3. training
  print("\n==> Training ...")
  for i in range(start_epoch, start_epoch+100):
    print("\nepoch", i)
    train(pcb, dataloader['train_all'], args.lr)
    if i % 5 == 0:
      mAP = test(pcb, dataloader, img_datasets)
      if mAP > best_mAP:
        best_mAP = mAP
        state = {
          'model': pcb,
          'mAP': best_mAP,
          'epoch': i
        }
        torch.save(state, checkpoint_path)
        print("==> checkpoint saved to", checkpoint_path, "with mAP", best_mAP)
