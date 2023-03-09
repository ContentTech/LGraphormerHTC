Implement of HMUDK: Hierarchy-aware Model Using Data Knowledge for Hierarchy Text Classification


# Requirements
Python >= 3.6
torch >= 1.6.0
transformers >= 4.11.0
datasets
torch-geometric == 1.7.2
torch-scatter == 2.0.8
torch-sparse == 0.6.12
Preprocess
Please download the original dataset and then use these scripts.

# Web Of Science
The original dataset can be acquired in the repository of HDLTex. Preprocessing code could refer to the repository of HiAGM and we provide a copy of preprocessing code here. Please save the Excel data file Data.xlsx in WebOfScience/Meta-data as Data.txt.

cd data/WebOfScience
python preprocess_wos.py
python data_wos.py

# NYT
The original dataset can be acquired here. Place the unzipped folder nyt_corpus inside data/nyt (or unzip nyt_corpus_LDC2008T19.tgz inside data/nyt).

cd data/nyt
python data_nyt.py

# RCV1-V2
The preprocessing code could refer to the repository of reuters_loader and we provide a copy here. The original dataset can be acquired here by signing an agreement. Place rcv1.tar.xz and lyrl2004_tokens_train.dat (can be downloaded here) inside data/rcv1.

cd data/rcv1
python preprocess_rcv1.py ./
python data_rcv1.py

Train  
usage: train.py [-h] [--lr LR] [--data DATA] [--batch BATCH] [--early-stop EARLY_STOP] [--device DEVICE] [----output_dir OUTPUT_IDR] --name NAME [--update UPDATE] [--model MODEL] [--wandb] [--arch ARCH] [--layer LAYER] [--graph GRAPH] [--prompt-loss]
                [--low-res] [--seed SEED]

optional arguments:   
  -h, --help                show this help message and exit   
  --lr LR					Start learning rate. Default: 3e-5.   
  --data {WebOfScience,nyt,rcv1} Dataset   
  --batch BATCH             Batch size   
  --early-stop EARLY_STOP   Epoch before early stop   
  --device DEVICE           cuda or cpu. Default: cuda   
  --output_dir OUTPUT_DIR   the dir to save the checkpoint   
  --name NAME               A name for different runs   
  --update UPDATE           Gradient accumulate steps   
  --wandb                   Use wandb for logging   
  --seed SEED               Random seed  
  
  
Checkpoints are in OUTPUT_DIR/checkpoints/DATA-NAME. Two checkpoints are kept based on macro-F1 and micro-F1 respectively (checkpoint_best_macro.pt, checkpoint_best_micro.pt)   

Example:

python train.py --name test --batch 16 --data nyt












