import torch 
import cv2

import numpy as np 
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt 

from utils import load_config, get_train_augs, get_valid_augs, train_fn, eval_fn, SegmentationDataset
from model import SegmentationModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# set device for training
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# load config file
config = load_config()

# load train files in dataframe
df = pd.read_csv(config['files']['CSV_FILE'])

train_df, valid_df = train_test_split(df, test_size = 0.2, random_state = 42)

trainset = SegmentationDataset(train_df, get_train_augs(config['model']['IMAGE_SIZE']))

validset = SegmentationDataset(valid_df, get_valid_augs(config['model']['IMAGE_SIZE']))

print(f"Size of Trainset : {len(trainset)}")
print(f"Size of Validset : {len(validset)}")

trainloader = DataLoader(trainset, batch_size=config['model']['BATCH_SIZE'], shuffle = True)
validloader = DataLoader(validset, batch_size=config['model']['BATCH_SIZE'])

print(f"Total n of batches in trainloader: {len(trainloader)}")
print(f"Total n of batches in validloader: {len(validloader)}")


model = SegmentationModel()
model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr = config['model']['LR'])

best_valid_loss = np.Inf

for i in tqdm(range(config['model']['EPOCHS'])):

  train_loss = train_fn(trainloader, model, optimizer, DEVICE)
  valid_loss = eval_fn(validloader, model, DEVICE)

  if valid_loss < best_valid_loss:
    torch.save(model.state_dict(), 'best_model.pt')
    print('SAVED-MODEL')
    best_valid_loss = valid_loss
  print(f"Epoch: {i+1} Train Loss: {train_loss} Valid Loss: {valid_loss}")