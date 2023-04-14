import os, pickle, torch, csv
import numpy as np

# transformer
d_model = 512  # Embedding Size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # dimension of K(=Q), V
n_heads = 8  # number of heads in Multi-Head Attention


cuda_seed = None  # 1024
cuda = 1

#################### pretrain model ########################################################
if cuda:
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

labels = ['notouch', 'scratch', 'rub', 'tickle', 'pat', 'constant', 'stroke']
lb_to_ix = {lb: ix for ix, lb in enumerate(labels)}
ix_to_lb = {ix: lb for ix, lb in enumerate(labels)}

endswith = '.pth'

