# DD-GCN
This repo is the official implementation for the paper ["DD-GCN: Directed Diffusion Graph Convolutional Network for Skeleton-based Human Action Recognition"](https://ieeexplore.ieee.org/document/10219780), ICME 2023.
### Pipeline
![image](pipeline.png) 
- DD-GCN (a) has ten STGC layers, and each layer contains two modules: CAGC (b) and STSE (d). After global average pooling, Softmax is utilized for action classification. CAGC is the unit of channel-wise correlation modeling and Graph Convolution (GC) with activity partition strategy (c). STSE employs a Multi-head Attention Mechanism (MSA) and Group Temporal Convolution (GTC) for synchronized spatio-temporal embedding.
# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

- We provide the dependency file of our experimental environment. You can install all dependencies by creating a new Anaconda virtual environment and running `pip install -r requirements.txt `
- Run `pip install -e torchlight` 

# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU RGB+D 60 Skeleton
- NTU RGB+D 120 Skeleton
- NW-UCLA

#### NTU RGB+D 60 and 120

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU RGB+D 60)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU RGB+D 120)
   3. Extract the above files to `./data/nturgbd_raw`

#### NW-UCLA

1. Download the dataset from [here](https://www.dropbox.com/s/10pcm4pksjy6mkq/all_sqe.zip?dl=0)
2. Move `all_sqe` to `./data/NW-UCLA`

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - NW-UCLA/
    - all_sqe
      ... # raw data of NW-UCLA
  - ntu/
  - ntu120/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
```

#### Generating Data

- Generate NTU RGB+D 60 or NTU RGB+D 120 dataset:

```
 cd ./data/ntu # or cd ./data/ntu120
 # Get skeleton of each performer
 python get_raw_skes_data.py
 # Remove the bad skeleton 
 python get_raw_denoised_data.py
 # Transform the skeleton to the center of the first frame
 python seq_transformation.py
```

# Training & Testing

### Training

- Change the config file depending on what you want.
```
- Example: training DDGCN on NTU RGB+D 60 cross subject with bone data
python main.py --config config/nturgbd-cross-subject/train.yaml  # default
```
- To train the model on joint, motion, and bone motion modalities, setting `bone` or `vel` arguments in the config file `train.yaml` or in the command line. 

### Testing
- To test the trained models saved in <work_dir>, run the following command:
```
python main.py --config <work_dir>/config.yaml --work-dir <work_dir> --phase test --save-score True --weights <work_dir>/xxx.pt --device 0
```

```
- Example: training DDGCN on NTU RGB+D 60 cross subject with bone data
python main.py --config config/nturgbd-cross-subject/test.yaml  # default
```

## Acknowledgements
This repo is based on [CTR-GCN](https://github.com/Uason-Chen/CTR-GCN). Thanks to the original authors for their work!

# Citation

Please cite this work if you find it useful.
```
  @INPROCEEDINGS{10219780,
    author={Li, Chang and Huang, Qian and Mao, Yingchi},
    booktitle={2023 IEEE International Conference on Multimedia and Expo (ICME)}, 
    title={DD-GCN: Directed Diffusion Graph Convolutional Network for Skeleton-based Human Action Recognition}, 
    year={2023},
    pages={786-791},
    doi={10.1109/ICME55011.2023.00140}
  }
```
