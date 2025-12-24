# Environment Requirements

## Required Versions
- **Python**: 3.9
- **CUDA**: 11.3

## Required Packages

```bash
torch==1.12.1 
torchvision==0.13.1 
torchaudio==0.12.1
torch-geometric==2.2.0 
torch-sparse==0.6.16 
torch-scatter==2.1.0 
torch-cluster==1.6.0 
torch-spline-conv==1.2.1
networkx==3.1 
pynauty==2.8.8.1 
tqdm 
pandas 
matplotlib 
seaborn 
scikit-learn==1.2.0 
numpy==1.23.5 
scipy==1.9.3 
ogb==1.3.6
```


## Running

To process graph datasets:
```bash
python Run_Graph.py
```
To process network datasets:
```bash
python Run_Net.py
```