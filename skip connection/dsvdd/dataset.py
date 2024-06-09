
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from random import sample
from random import shuffle


class DSVDD_dataset(Dataset):
   def __init__(self, data, label, mode):
      super().__init__()
      random.seed(42)
      self.data = data
      self.label = label
      self.patch_size = 17
      self.boundary = int(self.patch_size / 2)
      self.mode = mode
      

      normal_index = []
      anomal_index = []
      boundary = int(self.patch_size / 2)

      for i in range(boundary ,self.data.shape[0] - boundary): # 400
        for j in range(boundary ,self.data.shape[1] - boundary): # 512
            if self.label[i][j] == 0:
                continue
            elif self.label[i][j] == 2:
                normal_index.append((i, j))
            else:
                anomal_index.append((i, j))

      self.train_index = []
      self.valid_index = []
      self.test_index = []
      ratio = 0.4

      self.train_index += sample(normal_index, int(ratio * len(normal_index)))
      self.train_index += sample(anomal_index, int(ratio * len(anomal_index)))
      shuffle(self.train_index)

      self.test_index += normal_index
      self.test_index += anomal_index

      self.valid_index = [x for x in self.test_index if x not in self.train_index]
   
   def __len__(self):
      if self.mode == 'train':  
        return int(len(self.train_index))
      elif self.mode == 'valid':
        return int(len(self.valid_index))
      else:
        return int(len(self.test_index))
   
   def __getitem__(self, index):
      if self.mode == 'train':
        center_pixel = self.train_index[index]
      elif self.mode == 'valid':
        center_pixel = self.valid_index[index]
      else:
        center_pixel = self.test_index[index]
      img_pixel = self.data[center_pixel[0] - self.boundary:center_pixel[0] + self.boundary + 1, center_pixel[1] - self.boundary:center_pixel[1] + self.boundary + 1]
      img_pixel = np.float32(img_pixel)
      img_pixel = (img_pixel - np.min(img_pixel)) / (np.max(img_pixel) - np.min(img_pixel))
      img_pixel = torch.from_numpy(img_pixel)
      if self.label[center_pixel[0], center_pixel[1]] != 2:
         label_pixel = 0
      else:
         label_pixel = 1
      label_pixel = torch.as_tensor(label_pixel)

      return img_pixel.permute(2, 0, 1), label_pixel
   


class Indian_pines_dataset(Dataset):
    def __init__(self, img, gt, mode, patch_size, normal = 1):
        super().__init__()
        random.seed(42)
        self.img = img
        self.gt = gt
        self.normal = normal
        self.d = patch_size
        self.boundary = int(self.d / 2)
        self.h = self.img.shape[0]
        self.mode = mode
        gt, _ = np.unique(self.gt, return_counts = True)
        n_classes = len(gt)

        c_list = [[] for _ in range(n_classes - 1)]
        self.test_index = []

        for k in range(1, n_classes):
            for i in range(self.boundary, self.h - self.boundary):
                for j in range(self.boundary, self.h - self.boundary):
                    if self.gt[i][j] == k:
                        c_list[k - 1].append((i, j))
                        self.test_index.append((i, j))

        self.train_index = []
        self.train_index = c_list[self.normal - 1]
        # # valid_index = []

        # train_ratio = ratio
        # for i in range(1, n_classes):
        #     self.train_index += sample(c_list[i], k = int(len(c_list[i]) * (train_ratio / 100)))


        # for i in range(1, n_classes):
        #     valid_index += c_list[i]

        # valid_index = [x for x in valid_index if x not in train_index]
        

    def __len__(self):
        if self.mode == 'train':
          return len(self.train_index)
        else:
          return len(self.test_index)

    def __getitem__(self, index):
        if self.mode == 'train':
          center_pixel = self.train_index[index]
        else:
          center_pixel = self.test_index[index]

        img_pixel = self.img[center_pixel[0] - self.boundary:center_pixel[0] + self.boundary + 1, center_pixel[1] - self.boundary:center_pixel[1] + self.boundary + 1]
        img_pixel = (img_pixel - np.min(img_pixel)) / (np.max(img_pixel) - np.min(img_pixel))
        gt_pixel = self.gt[center_pixel[0], center_pixel[1]]
        if gt_pixel == self.normal:
           gt_pixel = 1
        else:
           gt_pixel = 0
        img_pixel = np.float32(img_pixel)
        img_pixel = torch.from_numpy(img_pixel)
        gt_pixel = torch.as_tensor(gt_pixel)
        return  img_pixel.permute(2, 0, 1), gt_pixel