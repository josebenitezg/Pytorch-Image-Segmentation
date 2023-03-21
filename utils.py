
import cv2
import torch
import yaml
import numpy as np
import albumentations as A
from torch.utils.data import Dataset


def get_train_augs(IMAGE_SIZE):
    
  return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.5)
  ])
  
def get_valid_augs(IMAGE_SIZE):
    
  return A.Compose([
      A.Resize(IMAGE_SIZE, IMAGE_SIZE),
  ])

def train_fn(data_loader, model, optimizer, DEVICE):

  model.train()
  total_loss = 0.0

  for images, masks in data_loader:

    images = images.to(DEVICE)
    masks = masks.to(DEVICE)

    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    return total_loss / len(data_loader)

def eval_fn(data_loader, model, DEVICE):

  model.eval()
  total_loss = 0.0
  with torch.no_grad():
    for images, masks in data_loader:

      images = images.to(DEVICE)
      masks = masks.to(DEVICE)

      logits, loss = model(images, masks)

      total_loss += loss.item()

  return total_loss / len(data_loader)

def load_config():
    config_file = f'config/config.yaml'

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    return config

  
class SegmentationDataset(Dataset):
  
  def __init__(self, df, augmentations):

    self.df = df
    self.augmentations = augmentations

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):

    row = self.df.iloc[idx]

    image_path = row.images
    mask_path = row.masks

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) #(h, w, c)
    # Resize the mask to the same dimensions as the image
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST) # (h, w)
    mask = np.expand_dims(mask, axis = -1)

    if self.augmentations:
      data = self.augmentations(image = image, mask = mask)
      image = data['image']
      mask = data['mask']

    # (h, w, c) -> (c, h, w)
    image = np.transpose(image, (2,0,1)).astype(np.float32)
    mask = np.transpose(mask, (2,0,1)).astype(np.float32)

    image = torch.Tensor(image) / 255.0
    mask = torch.round(torch.Tensor(mask) / 255.0)

    return image, mask
