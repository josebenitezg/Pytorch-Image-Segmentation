import torch 
import numpy as np
import gradio as gr
from model import SegmentationModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SegmentationModel()
model.to(DEVICE)
model.load_state_dict(torch.load('./best_model.pt'))

def inference(input_img):

    image = torch.from_numpy(input_img).permute(2,0,1).float()

    logits_mask = model(image.to(DEVICE).unsqueeze(0)) # (C, H, W) -> (1, C, H, W)
    pred_mask = torch.sigmoid(logits_mask)
    
    return pred_mask.squeeze().detach().cpu().numpy()

demo = gr.Interface(inference, gr.Image(shape=(224, 224)), "image")
demo.launch()