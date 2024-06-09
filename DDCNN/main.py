import os
import torch
from torch import nn
import numpy as np
from model import DDCNN
from scipy import io
from torch.utils.data import DataLoader, Dataset
import random
from torchmetrics import CohenKappa


indian_pine_corrected = io.loadmat('Indian_pines_corrected.mat')
indian_pine_corrected = indian_pine_corrected['indian_pines_corrected']
ip_c = indian_pine_corrected.shape[2]
print(indian_pine_corrected.shape)

indian_pine_gt = io.loadmat('Indian_pines_gt.mat')
indian_pine_gt = indian_pine_gt['indian_pines_gt']
gt, gt_class = np.unique(indian_pine_gt, return_counts = True)
n_classes = len(gt)
print(n_classes)


class hsi_dataset(Dataset):
    def __init__(self, img_dir, gt_dir, index_num, patch_size):
        super().__init__()
        img_mat = io.loadmat(img_dir)
        gt_mat = io.loadmat(gt_dir)
        self.img = img_mat['indian_pines_corrected']
        self.gt = gt_mat['indian_pines_gt']
        self.d = patch_size
        self.boundary = int(self.d / 2)
        self.img_shape = self.img.shape[0]
        self.index_num = index_num

    def __len__(self):
        return len(self.index_num)

    def __getitem__(self, index):
        center_pixel = self.index_num[index]
        img_pixel = self.img[center_pixel[0] - self.boundary:center_pixel[0] + self.boundary + 1, center_pixel[1] - self.boundary:center_pixel[1] + self.boundary + 1]
        gt_pixel = self.gt[center_pixel[0], center_pixel[1]]
        img_pixel = np.float32(img_pixel)
        img_pixel = torch.from_numpy(img_pixel)
        gt_pixel = torch.as_tensor(gt_pixel)
        return  img_pixel.permute(2, 0, 1), gt_pixel-1

patch_size = 9
batch_size = 100
boundary = int(patch_size / 2)
img_shape = 145
epochs = 100
device = torch.device('cuda:0')

c_list = [[] for _ in range(n_classes)]

for k in range(1, n_classes):
    for i in range(boundary, img_shape - boundary):
        for j in range(boundary, img_shape - boundary):
            if indian_pine_gt[i][j] == k:
                c_list[k].append((i, j))


train_index = []
valid_index = []
train_ratio = 15
for i in range(1, n_classes):
    train_index += random.sample(c_list[i], k = int(len(c_list[i]) * (train_ratio / 100)))

for i in range(1, n_classes):
    valid_index += c_list[i]

valid_index = [x for x in valid_index if x not in train_index]


train_dataset = hsi_dataset('Indian_pines_corrected.mat', 'Indian_pines_gt.mat', train_index, patch_size)
valid_dataset = hsi_dataset('Indian_pines_corrected.mat', 'Indian_pines_gt.mat', valid_index, patch_size)

dataloaders = {}
dataloaders['train'] = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
dataloaders['valid'] = DataLoader(valid_dataset, batch_size = batch_size, shuffle = True, drop_last = True)


model = DDCNN(n_classes, ip_c).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
criterion = nn.CrossEntropyLoss()
kappa = CohenKappa(task = 'multiclass', num_classes = n_classes - 1).to(device)

def save_model(model_state, model_name, save_dir = "./trained_model"):
    os.makedirs(save_dir, exist_ok = True)
    torch.save(model_state, os.path.join(save_dir, model_name))

for epoch in range(epochs + 1):
    losses = {}
    kappa_l = {}
    acc = {}

    best_valid_loss = 9999
    early_stop, early_stop_max = 0, 5

    training_loss = 0
    training_kappa = 0
    training_acc = 0

    for _, batch in enumerate(dataloaders['train']):
        images = batch[0].to(device)
        classes = batch[1].to(device)
        optimizer.zero_grad()
        model.train()

        pred = model(images)
        loss = criterion(pred, classes)        
        kappa_loss = kappa(pred, classes)  

        loss.backward()
        optimizer.step()

        pred = torch.argmax(pred, dim = 1)
        sub_acc = (pred == classes)

        training_acc += sub_acc.sum() / batch_size
        training_loss += loss.item()
        training_kappa += kappa_loss.item()

    losses['train'] = training_loss / len(dataloaders['train'])
    kappa_l['train'] = training_kappa / len(dataloaders['train'])
    acc['train'] = training_acc / len(dataloaders['train'])

    if epoch % 10 == 0:
        model.eval()

        valid_acc = 0
        valid_loss = 0
        valid_kappa = 0

        for _, batch in enumerate(dataloaders['valid']):
            images = batch[0].to(device)
            classes = batch[1].to(device)

            pred = model(images)
            loss = criterion(pred, classes)        
            kappa_loss = kappa(pred, classes)  

            pred = torch.argmax(pred, dim = 1)
            sub_acc = (pred == classes)
        
            valid_acc += sub_acc.sum() / batch_size
            valid_loss += loss.item()
            valid_kappa += kappa_loss.item()

        losses['valid'] = valid_loss / len(dataloaders['valid'])
        kappa_l['valid'] = valid_kappa / len(dataloaders['valid'])
        acc['valid'] = valid_acc / len(dataloaders['valid'])

        if best_valid_loss > valid_loss:
            best_valid_loss = valid_loss
            early_stop = 0

            print(f"{epoch}/{epochs} - train acc: {acc['train']}, train loss: {losses['train']}, val_acc: {acc['valid']}, val_loss: {losses['valid']}")
            print(f"{epoch}/{epochs} - train kappa: {kappa_l['train']}, val_kappa: {kappa_l['valid']}")
        else:
            early_stop += 1
            print(f"{epoch}/{epochs} - train acc: {acc['train']}, train loss: {losses['train']}, val_acc: {acc['valid']}, val_loss: {losses['valid']}")
            print(f"{epoch}/{epochs} - train kappa: {kappa_l['train']}, val_kappa: {kappa_l['valid']}")

    if early_stop >= early_stop_max:
        break

# best_model = copy.deepcopy(model.state_dict())
# save_model(best_model, "model.pth")



# ckpt = torch.load('./trained_model/model.pth')
# model = DDCNN(n_classes, ip_c).to(device)
# model.load_state_dict(ckpt)
# model.eval()

test_index = []
test_acc = 0

test_c_list = []
# for i in range(1, n_classes):
#     test_index += random.sample(c_list[i], k = int(len(c_list[i]) * (train_ratio / 100)))
# for i in range(1, n_classes):
    # test_c_list += c_list[i]
    # test_dataset = hsi_dataset('Indian_pines_corrected.mat', 'Indian_pines_gt.mat', c_list[i])

test_index += train_index
test_index += valid_index

test_dataset = hsi_dataset('Indian_pines_corrected.mat', 'Indian_pines_gt.mat', test_index, patch_size)

dataloaders['test'] = DataLoader(test_dataset, batch_size = 100, shuffle = True, drop_last = True)
class_per_acc = [[] for _ in range(n_classes)]
running_kappa = 0
output = []
ground_truth = []
acc = 0

with torch.no_grad():
    for _, batch in enumerate(dataloaders['test']):
        images = batch[0].to(device)
        classes = batch[1].to(device)

        pred = model(images)
    
        pred = torch.argmax(pred, dim = 1)
        # output.extend(torch.argmax(pred.cpu(), dim = 1).tolist())
        # ground_truth.extend(classes.cpu())
        sub_acc = (pred == classes)
        kappa_loss = kappa(pred, classes)  
        acc += sub_acc.sum() / batch_size
        running_kappa += kappa_loss.item()

        for i in range(len(pred)):
            if pred[i] == classes[i]:
                class_per_acc[classes[i]].append(1)
            else:
                class_per_acc[classes[i]].append(0)


    oa = acc / len(dataloaders['test']) * 100
    # oa = accuracy_score(output, ground_truth)
    kappa_l = running_kappa / len(dataloaders['test'])
    print('----------------------')
    # print(gt[i])
    print(oa)
    print(kappa_l)
    print('----------------------')

    for i in range(n_classes - 1):
        print(f'{i + 1} class : {sum(class_per_acc[i]) / len(class_per_acc[i]) * 100} %')

    