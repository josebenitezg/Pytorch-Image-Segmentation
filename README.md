# Pytorch Image Segmentation

## This repo contains the code for training a U-Net model for image segmentation on the Human Segmentation Dataset.

<a href="https://colab.research.google.com/github/josebenitezg/Pytorch-Image-Segmentation/blob/main/Learn_PyTorch_ImageSegmentation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"> 
</a>

## Usage :nut_and_bolt:

1. Clone this repo

```
git clone https://github.com/josebenitezg/Pytorch-Image-Segmentation
```

2. Create a virtual enviroment

```
python -m venv env
```

3. Activate virtual enviroment

- for linux

```
source env/bin/activate
```

- for windows

```
env\Scripts\Activate.bat
```

4. Install requirements

```
pip install -r requirements.txt
```

5. Train the model

```
python train.py
```

6. Run gradio inference app

```
python gradio_inference.py
```

This repo contains dataset files to train a small model.

Dataset Credit : https://github.com/VikramShenoy97/Human-Segmentation-Datasets