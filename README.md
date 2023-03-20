# DD-GCN
This repo is the official implementation for the paper "DD-GCN: Directed Diffusion Graph Convolutional Network for Skeleton-based Human Action Recognition", ICME 2023.
Codes will be available after publication as soon as possible.  
The pipeline of DD-GCN is shown in the following figure.
![image](https://github.com/shiyin-lc/pipelines/blob/main/pipeline-DD-GCN.png)
DD-GCN (a) has ten STGC layers, and each layer contains two modules: CAGC (b) and STSE (d). After global average pooling, Softmax is utilized for action classification. CAGC is the unit of channel-wise correlation modeling and Graph Convolution (GC) with activity partition strategy (c). STSE employs Multi-head Attention Mechanism (MSA) and Group Temporal Convolution (GTC) for synchronized spatio-temporal embedding.
