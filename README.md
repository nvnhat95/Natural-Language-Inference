# Natural-Language-Inference
This repository contains a PyTorch implementation for an LSTM and self attention models in SNLI dataset, which is a part of final project for LUS course. The self attention model follows this paper: A Decomposable Attention Model for Natural Language Inference https://arxiv.org/abs/1606.01933

## Project structure:
- folder *data*: contains SNLI dataset and embedding matrices
- folder *preprocessed*: contains data after preprocessing
- folder *save_model*: contains models after training
- file *preprocess.py*: code to preprocess data, including extracting data, loading word embedding matrix,...
- file *dataset.py*: a class to handle dataset for training
- file *model.py*: a class describing network architectures
- file *train.py*: contains function for training and testing procedures
- file *main.py*: main file to run program

## How to run
### Download data
Firstly you have to download data including SNLI data and Embedding matrices:

Download the SNLI dataset from https://nlp.stanford.edu/projects/snli/snli_1.0.zip and GLoVe matrices from http://nlp.stanford.edu/data/glove.6B.zip and http://nlp.stanford.edu/data/glove.42B.300d.zip then extract all of them into folder *data* 

If you don't want to do it manually, I also provided script to download and extract data, just run the following command:
```
python main.py --download_data
```
### Preprocessing
To preprocess data, you should determine which word embedding matrix you want to use. In default, with the data you have downloaded above, you can choose embedding matrices in one of {6B.50d, 6B.100d, 6B.200d, 6B.300d, 42B.300d} when running:
```
python main.py --preprocess_data --embedding=6B.300d
```
This step will takes several minutes. If you want to use other embedding matrices, download from https://nlp.stanford.edu/projects/glove/ and repeat above steps.

After preprocessing, all the preprocessed files will be in folder *preprocessed*, which consists of:
- files *premise_.txt, hypothesis_.txt, label_.txt*: data in text format, each line corresponds to 1 sample.
- files *word.dict*, *label.dict* and *POS.dict*: dictionaries for tokens in training set, label of classes, POS tags respectively.
- files *train.hdf5*, *test.hdf5*, *test.hdf5*: 3 sets in the hdf5 format.

### Train models
The main arguments of program are:
- use_POS: use POS tags as an additional feature
- model_type: which kind of model to use: ['attention', 'lstm', 'combine']
- hidden_dim: size of hidden dimension, default 200
- learning_rate: default 2e-4
- dropout_rate: default 0.2
- max_epochs: maximum epoch to train, default 100
- gpu: use GPU to train

An example for training a self attention network, use POS tags, on GPU:
```
python main.py --gpu --model_type=attention --use_POS --hidden_dim=200 --max_epochs=100
```

## Licence
MIT
