# LSG-GMCR: Large-Scale Group Conflict Resolution

This repository contains the code and anonymized dataset for our paper submitted to European Journal of Operational Research.

## Repository Structure

- `data/`: Anonymized Weibo dataset (4,724 entries)
- `code/`: Python implementation of FCM-DBSCAN meta-clustering and LSG-GMCR

## Dependencies

This project requires Python 3.8+ and CUDA-capable GPU (optional but recommended).

### Installation

```bash
# Using pip
pip install -r requirements.txt

# Using conda (recommended for GPU support)
conda env create -f environment.yml
conda activate lsg-gmcr

This requirements.txt covers all the imports found in your LSTM+CNN+Word2Vec code and should work with your other scripts as well.
