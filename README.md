# PBMC: Patch Based Material Characterization
This repository is a framework for the registration and segmentation of image pairs using patch-based convolutional neural networks

<p align="center">
  <img src="https://github.com/user-attachments/assets/e81814f3-47e5-4848-9e6c-51c2254e4fc3" width="300" />
  <img src="https://github.com/user-attachments/assets/40d24ad3-9aa6-4d39-80da-8a349df7b1e1" width="300" /> 
  <img src="https://github.com/user-attachments/assets/29662572-6c8b-483b-b943-ed51f065b8dd" width="300" />
  <img src="https://github.com/user-attachments/assets/d8d5f17c-fafd-44ff-adec-5dcaf18a697d" width="300" />
  <img src="https://github.com/user-attachments/assets/5eaadd4e-3de1-4925-8dbc-2e3682ec7b84" width="300" />
</p>


## Setup
```python
cd PBMC
conda env create --name pbmc --file=environments.yml
```

## Recreate Our Results
```python
cd Internship\ Project/PBMC/
conda activate tensorflow2.16.1
python main.py -model unet -augment -epochs 300 -gpu 0
```

```python
cd Internship\ Project/PBMC/
conda activate tensorflow2.16.1
python main.py -model attention -augment -epochs 300 -gpu 1
```

```python
cd Internship\ Project/PBMC/
conda activate tensorflow2.16.1
python main.py -model residual -augment -epochs 300 -gpu 2
```

## Train PBCNN Models
```python
cd script
# Train Patch-Based U-Net
python main.py -model unet -epochs 100 -gpu 0
# Train Patch-Based Attention U-Net
python main.py -model attention -epochs 100 -gpu 0
# Train Patch-Based Residual U-Net
python main.py -model residual -epochs 100 -gpu 0
```

## Cite this work
Wang et al, [*Fine Pore Segmentation with Deep Neural Networks*](https://www.nature.com/articles/s41598-023-48800-3), Scientific Reports 2023.
```
@article{wang2023fine,
  title={A fine pore-preserved deep neural network for porosity analytics of a high burnup U-10Zr metallic fuel},
  author={Wang, Haotian and Xu, Fei and Cai, Lu and Salvato, Daniele and Di Lemma, Fidelma Giulia and Capriotti, Luca and Yao, Tiankai and Xian, Min},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={22274},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

