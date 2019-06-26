import os, wget, subprocess, argparse
import numpy as np
import torch
import torch.optim as optim
import h5py
from dataset import SNLI_data
from model import Model
import train

def download_data(glove='glove.6B.zip'):
    print("Downloading word embedding...")
    downloaded_glove1 = wget.download("http://nlp.stanford.edu/data/{}".format('glove.6B.zip'))
    downloaded_glove2 = wget.download("http://nlp.stanford.edu/data/{}".format('glove.42B.300.zip'))
    print("Downloading SNLI dataset...")
    downloaded_snli = wget.download("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
    
    if not os.path.exists("./data"):
        os.mkdir("./data")
    print("Extracting...")
    zip = zipfile.ZipFile(downloaded_glove1)
    zip.extractall(path="./data")
    zip = zipfile.ZipFile(downloaded_glove2)
    zip.extractall(path="./data")
    zip = zipfile.ZipFile(downloaded_snli)
    zip.extractall(path="./data")
    print("done!")


def parse_args():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--download_data', action='store_true', help="download mode")
    parser.add_argument('--data_folder', help="location of folder with the preprocessed data")
    parser.add_argument('--preprocess_data', action='store_true', help="preprocess data")
    parser.add_argument('--embedding', choices=['6B.50d', '6B.100d', '6B.200d', '6B.300d', '42B.300d'], \
                        help="type of word embedding_matrix, one in ['6B.50d', '6B.100d', '6B.200d', '6B.300d', '42B.300d']")
    parser.add_argument('--use_POS', action='store_true', help="use POS tag feature")
    parser.add_argument('--model_type', choices=['attention', 'lstm', 'combine'], default='attention', \
                                        help="type of model to use: ['attention', 'lstm', 'combine']")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden dimension")
    parser.add_argument('--learning_rate', type=float, default=0.0002, help="learning rate")
    parser.add_argument('--dropout_rate', type=float, default=0.2, help="dropout rate")
    parser.add_argument('--max_epochs', type=int, default=100, help="maximum epochs")
    parser.add_argument('--gpu', action='store_true', help="use gpu to train")
    args = parser.parse_args()
    return args
    
    
if __name__=='__main__':
    args = parse_args()
    if args.download_data:
        download_data()
        exit(0)
    
    if args.preprocess_data:
        assert args.embedding is not None
        subprocess.call("python preprocess.py --data_folder=./data/snli_1.0 \
                        --glove=./data/glove.{}.txt --seqlength=100".\
                        format(args.embedding))
        exit(0)
    
    f = h5py.File("./preprocessed/glove.hdf5", 'r')
    wordvec_matrix = torch.from_numpy(np.array(f['word_vecs'], dtype=np.float32))
    
    use_padding = False
    train_data = SNLI_data("./preprocessed/train.hdf5", use_padding=use_padding)
    dev_data = SNLI_data("./preprocessed/dev.hdf5", use_padding=use_padding)
    test_data = SNLI_data("./preprocessed/test.hdf5", use_padding=use_padding)
    
    POS_embedding = None
    if args.use_POS:
        POS_embedding = torch.eye(int(train_data.POS_size.value))
    
    # set the device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Found device: ", device)
        
    model = Model(embedding_matrix=wordvec_matrix, hidden_dim=args.hidden_dim, \
              feature_extractor=args.model_type, dropout_rate=args.dropout_rate, POS_embedding=POS_embedding).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    if not os.path.exists("./saved_model"):
        os.mkdir("./saved_model")
    model_path = "./saved_model/model.pt"
    
    train.run(model=model,  \
              optimizer=optimizer, \
              train_data=train_data, \
              dev_data=dev_data, \
              test_data=test_data, \
              max_epochs=args.max_epochs, \
              device=device, \
              model_path=model_path)