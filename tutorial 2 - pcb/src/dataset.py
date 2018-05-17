import os
import torch
from torchvision import datasets, transforms
opj = os.path.join

from config import Config
from shutil import copyfile
from utils import make_dir


def transform_dataset(src_path, dst_path):
  """
  Re-organize dataset with id

  Args:
    src_path: Path to (raw) dataset directory
    dst_path: Path to transformed dataset   
  """

  config = Config()
  name_mapping = config('dir_mapping')

  for img_type in os.listdir(src_path):
    if img_type in name_mapping.keys():
      src_img_dir = opj(src_path, img_type)
      dst_img_dir = opj(dst_path, name_mapping[img_type])
      for img_name in os.listdir(src_img_dir):
        id = img_name.split('_')[0]  # id_camera&sequence_frame_bbox
        src_img = opj(src_img_dir, img_name)
        dst_img = opj(dst_img_dir, id, img_name)
        make_dir(opj(dst_img_dir, id))
        copyfile(src_img, dst_img)


def make_val(path):
  """Make validation sets
  Args:
    path: path to transformed images root folder
  """

  all_img_dir = opj(path, 'train_all')
  train_img_dir = opj(path, 'train')
  val_img_dir = opj(path, 'val')

  for id in os.listdir(all_img_dir):
    all_imgs = os.listdir(opj(all_img_dir, id))
    make_dir(opj(train_img_dir, id))
    make_dir(opj(val_img_dir, id))
    for i in range(len(all_imgs)):
      img_name = all_imgs[i]
      if i == 0:
        src_img = opj(all_img_dir, id, img_name)
        dst_img = opj(val_img_dir, id, img_name)
      else:
        dst_img = opj(train_img_dir, id, img_name)
      copyfile(src_img, dst_img)


def prepare_dataset(path, batch_size):
  """prepare (transformed) dataset for training/testing

  Args:
    path: path to (transformed) dataset
    batch_size: batch size

  Return:
    dataloader: Dataloader dictionary.
      Keys are 'train', 'gallery' and 'query' 
  """

  transform_train_list = [
    transforms.Resize(size=(384, 128), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]

  transform_test_list = [
    transforms.Resize(size=(384, 128), interpolation=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
  ]

  train_transform = transforms.Compose(transform_train_list)
  test_transform = transforms.Compose(transform_test_list)

  train_datasets = {x: datasets.ImageFolder(opj(path, x), transform=train_transform)
                    for x in ['train_all', 'train']}
  test_datasets = {x: datasets.ImageFolder(opj(path, x), transform=test_transform)
                   for x in ['val', 'gallery', 'query']}
  img_datasets = dict(list(train_datasets.items()) +
                      list(test_datasets.items()))

  train_loader = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=batch_size,
                                                 num_workers=4, shuffle=True)
                  for x in ['train_all', 'train']}
  test_loader = {x: torch.utils.data.DataLoader(img_datasets[x], batch_size=batch_size,
                                                num_workers=4)
                 for x in ['val', 'gallery', 'query']}
  dataloader = dict(list(train_loader.items()) + list(test_loader.items()))

  return img_datasets, dataloader
