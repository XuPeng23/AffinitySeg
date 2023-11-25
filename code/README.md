# Label-aware Attention Network with Multi-scale Boosting for Medical Image Segmentation
This is the official implementation for: **Label-aware Attention Network with Multi-scale Boosting for Medical Image Segmentation**



## Dataset

  + Download the dataset from:

    Google Drive: https://drive.google.com/file/d/1-qdkjv3UMancVAoP80V6zKPwqrkah-Kn/view?usp=drive_link

    Baidu Netdisk：https://pan.baidu.com/s/1b-QjjGAv9VQpEJRnYCii4w?pwd=wcm6 
    password：wcm6

  + Dataset is ordered as follow:
```
|-- data
|   |-- GlaS
|   |   |-- train
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- test
|   |   |   |-- images
|   |   |   |-- masks
|   |-- TNBC
|   |   |-- train
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- test
|   |   |   |-- images
|   |   |   |-- masks
|   |-- GlaS
|   |   |-- fold1
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- fold2
|   |   |   |-- images
|   |   |   |-- masks
|   |   |-- fold3
|   |   |   |-- images
|   |   |   |-- masks
```

## Training

+ Run train_PanNuke.py to train model on the PanNuke dataset.
+ Run train_5_cross_validation_GlaS/TNBC to train model on the GlaS dataset or the TNBC dataset.

Two folders named log and model will store logging information and the trained model.

+ Well-trained models can be downloaded from

  Google Drive: https://drive.google.com/file/d/1rE3ialHQkHt9ZUYswFmbPwy0uwbsNwI2/view?usp=drive_link

  Baidu Netdisk: https://pan.baidu.com/s/19uY9gc_U46X8MEcR0v34tw?pwd=xjfv 
  Password: xjfv

## Environment

The code is developed on one NVIDIA RTX 4090 GPU with 24 GB memory and tested in Python 3.7.

pytorch  1.8.0
torchaudio  0.8.0
torchvision  0.13.1
numpy  1.21.5
timm  0.5.4
scipy  1.7.3
scikit-learn  0.24.2

## Citation

**If you think our work is helpful, please cite **

