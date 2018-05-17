import os
import sys
import platform

import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import average_precision_score


def make_dir(path):
  os.makedirs(path, exist_ok=True)


def make_cuda(x):
  return x.cuda()


def one_hot(x, length):
  batch_size = x.size(0)
  x_one_hot = torch.zeros(batch_size, length)
  for i in range(batch_size):
    x_one_hot[i, x[i]] = 1.0
  return x_one_hot
    

def get_cam_label(img_path):
  """Get camera number list and label list in one folder"""
  camera_ids = []
  labels = []
  for path, _ in img_path:
    if 'Windows' in platform.platform():  # for Windows
      filename = path.split('\\')[-1]
    else:  # for Linux or MacOS
      filename = path.split('/')[-1]
    label = filename[0:4]
    camera = filename.split('c')[1]
    if label[0:2] == '-1':
      labels.append(-1)
    else:
      labels.append(int(label))
    camera_ids.append(int(camera[0]))
  return np.array(camera_ids), np.array(labels)


def extract_feature(model, img):
  """Extract feature maps from model"""

  with torch.no_grad():
    _ = model(Variable(img.cuda()))
  
  feature = model.features_H.data.cpu().squeeze()
  feature = feature.view(feature.size(0), -1)  # [N, num_classes]

  return feature


def evaluate(query_features, query_labels, query_cams, gallery_features, gallery_labels, gallery_cams, progress_bar):
  CMC = torch.IntTensor(len(gallery_labels)).zero_()
  AP = 0
  for i in range(len(query_labels)):
    query_feature = query_features[i]
    query_label = query_labels[i]
    query_cam = query_cams[i]

    score = np.dot(gallery_features, query_feature)

    match_query_index = np.argwhere(gallery_labels == query_label)
    same_camera_index = np.argwhere(gallery_cams == query_cam)

    # Positive index is the matched indexs at different camera i.e. the desired result
    positive_index = np.setdiff1d(match_query_index, same_camera_index, assume_unique=True)

    # Junk index is the indexs at the same camera or the unlabeled image
    junk_index = np.append(
      np.argwhere(gallery_labels == -1),
      np.intersect1d(match_query_index, same_camera_index)
    )

    index = np.arange(len(gallery_labels))
    # Remove all the junk indexs
    sufficient_index = np.setdiff1d(index, junk_index)

    # compute AP
    y_true = np.in1d(sufficient_index, positive_index)
    y_score = score[sufficient_index]
    AP += average_precision_score(y_true, y_score)

    # Compute CMC
    # Sort the sufficient index by their scores, from large to small
    lexsort_index = np.argsort(y_score)
    sorted_y_true = y_true[lexsort_index[::-1]]
    match_index = np.argwhere(sorted_y_true == True)

    if match_index.size > 0:
      first_match_index = match_index.flatten()[0]
      CMC[first_match_index:] += 1

    progress_bar.move(leftmsg="evaluate", rightmsg="mAP %.4f" % (AP / (i+1)))

  CMC = CMC.float()
  CMC = CMC / len(query_labels)
  mAP = AP / len(query_labels)

  return CMC, mAP



class ProgressBar:
  def __init__(self, count=0, total=0, width=40):
    self.count = count
    self.total = total
    self.width = width

  def move(self, leftmsg, rightmsg):
    self.count += 1
    progress = int(self.width * self.count / self.total)
    sys.stdout.write(leftmsg + ' [ ')
    sys.stdout.write('#' * progress + '-' * (self.width - progress))
    sys.stdout.write(' ] ' + rightmsg + ' ' + '\r')

    if progress == self.width:
      sys.stdout.write('\n')

