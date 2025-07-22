# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraphPart is a neural graph partitioning system that uses Variational Graph Autoencoders (VGAE) for hypergraph partitioning. The system combines deep learning with traditional partitioning tools to optimize hypergraph cuts while maintaining balance constraints.

## Core Architecture

### Key Components

- **models.py**: Contains the main neural network architectures
  - `GraphPartitionModel`: VGAE-based model extending PyTorch Geometric's VGAE
  - `VariationalEncoder`: Hypergraph convolution encoder with masking
  - `PartitionDecoder`: Maps latent representations to partition assignments
  - `HyperData`: Custom data structure for hypergraph representation

- **utils.py**: Core preprocessing and evaluation utilities
  - Hypergraph incidence matrix normalization
  - Clique expansion graph creation
  - Topological feature computation using SVD/eigendecomposition
  - Integration with external partitioners (hMETIS, EasyPart, KaHyPar)
  - Multiprocessing evaluation framework

- **train.py**: Training script with custom loss combining KL divergence, normalized cut, and balance terms
- **test.py**: Inference script with iterative refinement using V-cycles
- **script.sh**: Batch evaluation script for multiple hypergraph instances

### Data Flow

1. Hypergraph files (.hgr format) are loaded from `data/` directory
2. Preprocessing creates 7-dimensional node features:
   - Clique expansion spectral features (2D)
   - Star expansion spectral features (2D)  
   - Node degree and pin count
   - Initial partition from hMETIS
3. Model performs variational encoding and partition decoding
4. External refinement tools improve partition quality
5. Evaluation uses hyperedge cut counting

## Development Commands

### Training
```bash
python train.py --lr 5e-5 --alpha 0.0005 --beta 5 --gamma 1 --epochs 50
```

### Testing Single Instance
```bash
python test.py --filename ibm02.hgr --modelname model.pt --num_partitions 2
```

### Batch Evaluation
```bash
bash script.sh
```

## External Dependencies

The system integrates with several external partitioning tools located in `exec/`:
- `hmetis2.0pre1`: Used for initial partition generation
- `EasyPart`: Post-processing refinement tool
- `KaHyPar`: Alternative refinement option (configurable)
- `cut_kKaHyPar_sea20.ini`: KaHyPar configuration file

## Key Implementation Details

- Uses normalized hypergraph incidence matrices for spectral features
- Implements custom hyperedge cut loss for hypergraph-aware training
- Supports both clique expansion and star expansion graph representations
- Features multiprocessing for parallel partition evaluation
- Model sampling generates multiple partition candidates per iteration