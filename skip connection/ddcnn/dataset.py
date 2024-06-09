import numpy as np
import torch
from torch.utils.data import Dataset
from random import sample
from random import shuffle

class DDCNN_dataset(Dataset):
   def __init__(self, data, label, mode):
      self.data = data
      self.label = label
      self.patch_size = 9
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
      ratio = 1

      self.train_index += sample(normal_index, int(0.7 * len(normal_index)))
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
      # img_pixel = self.data[center_pixel[0], center_pixel[1]]
      # img_pixel = img_pixel / np.max(img_pixel)
      img_pixel = np.float32(img_pixel)
      img_pixel = torch.from_numpy(img_pixel)
      if self.label[center_pixel[0], center_pixel[1]] != 2:
         label_pixel = 0
      else:
         label_pixel = 1
    #   label_pixel = self.label[center_pixel[0], center_pixel[1]]
    #   if label_pixel == 24:
    #      label_pixel = 15
    #   elif label_pixel == 25:
    #      label_pixel = 16
      label_pixel = torch.as_tensor(label_pixel)

      return img_pixel.permute(2, 0, 1), label_pixel