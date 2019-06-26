import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Module(nn.Module):
    def __init__(self, hidden_dim, num_layers=2, bidirectional=True, dropout_rate=0.0):
        super(LSTM_Module, self).__init__()
        self.num_layers = num_layers
        self.num_directions = bidirectional + 1
        self.hidden_dim = hidden_dim
        
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim * self.num_directions * self.num_layers * 2, hidden_dim)
        
    def forward(self, a, b):
        _, (_, v1) = self.LSTM(a)
        _, (_, v2) = self.LSTM(b)
        
#         v1 = torch.squeeze(v1, dim=0)
#         v2 = torch.squeeze(v2, dim=0)
        v1 = v1.permute(1, 0, 2).contiguous().view((-1, self.hidden_dim * self.num_directions * self.num_layers))
        v2 = v2.permute(1, 0, 2).contiguous().view((-1, self.hidden_dim * self.num_directions * self.num_layers))
        
        v1_cat_v2 = torch.cat((v1, v2), dim=1) # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.fc(v1_cat_v2)
        h = F.relu(h)
        
        return h
        
        
        
class SelfAttention_Module(nn.Module):
    def __init__(self, hidden_dim, use_BN=False, dropout_rate=0.0):
        super(SelfAttention_Module, self).__init__()
        def MLP(input_dim, output_dim, use_BN, dropout_rate):
            layers = []
            layers.append(nn.Linear(input_dim, output_dim))
            if use_BN:
                layers.append(nn.BatchNorm1d(output_dim, affine=True))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))

            layers.append(nn.Linear(output_dim, output_dim))
            layers.append(nn.ReLU())

            mlp = nn.Sequential(*layers)

            return mlp
        
        self.hidden_dim = hidden_dim
        self.F = MLP(hidden_dim, hidden_dim, use_BN, dropout_rate)
        self.G = MLP(hidden_dim * 2, hidden_dim, use_BN, dropout_rate)
        self.H = MLP(hidden_dim * 2, hidden_dim, use_BN, dropout_rate)
        self.LSTM = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        
    def forward(self, a, b):
        l_a = a.shape[1]
        l_b = b.shape[1]
        
        a = self.F(a.view(-1, self.hidden_dim)) # a: ((batch_size * l_a) x hidden_dim)
        a = a.view(-1, l_a, self.hidden_dim) # a: (batch_size x l_a x hidden_dim)
        b = self.F(b.view(-1, self.hidden_dim))  # b: ((batch_size * l_b) x hidden_dim)
        b = b.view(-1, l_b, self.hidden_dim) # b: (batch_size x l_b x hidden_dim)
        
        # equation (1) in paper:
        e = torch.bmm(a, torch.transpose(b, 1, 2)) # e: (batch_size x l_a x l_b)
        
        # equation (2) in paper:
        beta = torch.bmm(F.softmax(e, dim=2), b)  # beta: (batch_size x l_a x hidden_dim)
        alpha = torch.bmm(F.softmax(torch.transpose(e, 1, 2), dim=2), a) # alpha: (batch_size x l_b x hidden_dim)
        
        
        # equation (3) in paper:
        a_cat_beta = torch.cat((a, beta), dim=2)
        b_cat_alpha = torch.cat((b, alpha), dim=2)
        v1 = self.G(a_cat_beta.view(-1, 2 * self.hidden_dim)) # v1: ((batch_size * l_a) x hidden_dim)
        v2 = self.G(b_cat_alpha.view(-1, 2 * self.hidden_dim)) # v2: ((batch_size * l_b) x hidden_dim)
        
        
        
#         _, (_, v1) = self.LSTM(v1.view(-1, l_a, self.hidden_dim))
#         _, (_, v2) = self.LSTM(v2.view(-1, l_b, self.hidden_dim))
#         v1 = torch.squeeze(v1, dim=0)
#         v2 = torch.squeeze(v2, dim=0)

        
        # equation (4) in paper:
        v1 = torch.sum(v1.view(-1, l_a, self.hidden_dim), dim=1) # v1: (batch_size x 1 x hidden_dim)
        v2 = torch.sum(v2.view(-1, l_b, self.hidden_dim), dim=1) # v2: (batch_size x 1 x hidden_dim)
        
        v1 = torch.squeeze(v1, dim=1)
        v2 = torch.squeeze(v2, dim=1)
        
        # equation (5) in paper:
        v1_cat_v2 = torch.cat((v1, v2), dim=1) # v1_cat_v2: (batch_size x (hidden_dim * 2))
        h = self.H(v1_cat_v2)
        
        return h
        
        
class Model(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim, feature_extractor='attention', use_BN=False, dropout_rate=0.0, POS_embedding=None):
        super(Model, self).__init__()
        # embedding
        self.embedding_dim = embedding_matrix.shape[1]
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix)
        self.POS_embedding = POS_embedding
        if POS_embedding is not None:
            self.num_POS = POS_embedding.shape[1]
            self.POS_embedding = nn.Embedding.from_pretrained(POS_embedding)    
        else:
            self.num_POS = 0
        
        self.feature_extractor = feature_extractor
        assert feature_extractor in ['attention', 'lstm', 'combine']
        if feature_extractor == 'self_attention':
            self.feature_extractor_module = SelfAttention_Module(hidden_dim, use_BN, dropout_rate)
        else:
            self.feature_extractor_module = LSTM_Module(hidden_dim, dropout_rate=dropout_rate)
        
        
        self.LSTM = nn.LSTM(input_size=self.embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        
        
        # linear transformation from embedding
        self.input_fc = nn.Linear(self.embedding_dim + self.num_POS, hidden_dim, bias=True)
        # linear transformation to prediction
        self.output_fc = nn.Linear(hidden_dim, 3, bias=True)
        
    def forward(self, a, b, a_POS=None, b_POS=None):
        l_a = a.shape[1]
        l_b = b.shape[1]
        
        a = self.embedding(a) # a: (batch_size x l_a x embedding_dim)
        b = self.embedding(b) # b: (batch_size x l_b x embedding_dim)
        if self.POS_embedding is not None:
            try:
                a_POS = self.POS_embedding(a_POS)
                a = torch.cat((a, a_POS), dim=-1)
                b_POS = self.POS_embedding(b_POS)
                b = torch.cat((b, b_POS), dim=-1)
            except Exception as e:
                print(e)
        
        
        if self.feature_extractor is not 'combine':
            a = self.input_fc(a.view(-1, self.embedding_dim + self.num_POS))
            b = self.input_fc(b.view(-1, self.embedding_dim + self.num_POS))
            a = a.view(-1, l_a, self.hidden_dim) # a: (batch_size x l_a x hidden_dim)
            b = b.view(-1, l_b, self.hidden_dim) # b: (batch_size x l_b x hidden_dim)
        else:
            a, (_, _) = self.LSTM(a)
            b, (_, _) = self.LSTM(b)
            a = a.contiguous().view(-1, l_a, self.hidden_dim)
            b = b.contiguous().view(-1, l_b, self.hidden_dim)
        
        
        h = self.feature_extractor_module(a, b)
        
        y_hat = self.output_fc(h)
        
        return y_hat