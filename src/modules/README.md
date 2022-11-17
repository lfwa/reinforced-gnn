## Module Definitions
This directory specifies the models with the corresponding hyperparameters. Each of the models is contained in a single file. The default hyperparameters were tuned to be optimal for each of the models.

## Structure of Directory
- `gnn_baseline.py`: This model describes a Graph Neural Network based model based on the paper [Neural Graph Collaborative Filtering](https://arxiv.org/pdf/1905.08108.pdf).
- `gnn_ncf.py`: This model describes a Graph Neural Network model enhanced with the feed forward network of the Neural Collaborative Filtering model.
- `ncf_baseline.py`: This model describes the feed forward network of the Neural Collaborative Filtering model described in the paper [Neural Collaborative Filtering](https://arxiv.org/pdf/1708.05031.pdf).
- `nmf.py`: This model is a wrapper class of the Non-Negative Matrix Factorization implemented in the [Surprise Library](https://github.com/NicolasHug/Surprise).
- `reinforced_gnn_ncf.py`: This model describes the Graph Neural Network model enhanced with the feed forward network of the Neural Collaborative Filtering model and the reinforcements predicted by other models.
- `slopeone.py`: This model is a wrapper class of SlopeOne implemented in the [Surprise Library](https://github.com/NicolasHug/Surprise).
- `svd_unbiased.py`: This model is a wrapper class of the Singular Value Decomposition implemented in the [Surprise Library](https://github.com/NicolasHug/Surprise).
- `svdpp.py`: This model is a wrapper class of the Singular Value Decomposition++ implemented in the [Surprise Library](https://github.com/NicolasHug/Surprise).
