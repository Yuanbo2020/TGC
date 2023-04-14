import numpy as np
import torch
import torch.nn as nn
import os, math
import framework.config as config



class ScaledDotProductAttention_nomask(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention_nomask, self).__init__()

    def forward(self, Q, K, V, d_k=config.d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention_nomask(nn.Module):
    def __init__(self, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads,
                 output_dim=config.d_model):
        super(MultiHeadAttention_nomask, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layernorm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(n_heads * d_v, output_dim)

    def forward(self, Q, K, V, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        context, attn = ScaledDotProductAttention_nomask()(q_s, k_s, v_s)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class EncoderLayer(nn.Module):
    def __init__(self, output_dim=config.d_model):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention_nomask(output_dim=output_dim)

    def forward(self, enc_inputs):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, input_dim, n_layers, output_dim=config.d_model):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(output_dim) for _ in range(n_layers)])
        self.mel_projection = nn.Linear(input_dim, config.d_model)

    def forward(self, enc_inputs):
        # print(enc_inputs.size())  # torch.Size([64, 54, 8, 8])
        size = enc_inputs.size()
        enc_inputs = enc_inputs.view(size[0], size[1], -1)
        enc_outputs = self.mel_projection(enc_inputs)
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, d_k=config.d_k):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.fc = nn.Linear(n_heads * d_v, d_model)
        self.layernorm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask,
                d_model=config.d_model, d_k=config.d_k, d_v=config.d_v, n_heads=config.n_heads):
        residual, batch_size = Q, Q.size(0)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1,2)
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1,2)

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)

        output = self.fc(context)
        x = self.layernorm(output + residual)
        return x, attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model=config.d_model, d_ff=config.d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layerNorm = nn.LayerNorm(d_model)
    #     elementwise_affine就是公式中的γ \gammaγ和β \betaβ，前者开始为1，后者为0，二者均可学习随着训练过程而变化
    # https://blog.csdn.net/weixin_41978699/article/details/122778085

    def forward(self, inputs):
        residual = inputs
        inputs = inputs.transpose(1, 2)
        output = nn.ReLU()(self.conv1(inputs))
        output = self.conv2(output)
        output = output.transpose(1, 2)
        residual_ouput = self.layerNorm(output + residual)
        return residual_ouput



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def get_padding_mask(seq_providing_expand_dim, seq_with_padding):
    batch_size, len_q = seq_providing_expand_dim.size()
    batch_size, len_k = seq_with_padding.size()
    pad_attn_mask = seq_with_padding.data.eq(0).unsqueeze(1)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_padding_mask_with_acoustic_feature(seq_providing_expand_dim, seq_with_padding):
    batch_size, len_q, _ = seq_providing_expand_dim.size()
    batch_size, len_k = seq_with_padding.size()
    pad_attn_mask = seq_with_padding.data.eq(0).unsqueeze(-1)
    return pad_attn_mask.expand(batch_size, len_k, len_q)


def get_attn_subsequent_mask(seq):
    device = seq.device
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequent_mask = np.triu(np.ones(attn_shape), k=1)
    subsequent_mask = torch.from_numpy(subsequent_mask).bool()
    return subsequent_mask.to(device)



def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)



class Transformer_context(nn.Module):
    def __init__(self, input_dim, window, ntoken, encoder_layers, d_model):
        super(Transformer_context, self).__init__()
        self.encoder = Encoder(input_dim=input_dim, n_layers=encoder_layers, output_dim=config.d_model)

        self.fc1 = nn.Linear(d_model, 128)
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 4)
        self.fc4 = nn.Linear(window * 4, ntoken)

        self.bn0 = nn.BatchNorm2d(window)

    def init_weight(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_layer(self.fc4)

    def forward(self, enc_inputs):
        enc_inputs = self.bn0(enc_inputs)

        enc_outputs, enc_self_attns = self.encoder(enc_inputs)

        enc_outputs = torch.relu(self.fc1(enc_outputs))

        enc_outputs = torch.relu(self.fc2(enc_outputs))

        enc_outputs = torch.relu(self.fc3(enc_outputs))

        enc_outputs = enc_outputs.view(len(enc_outputs), -1)

        dec_logits = self.fc4(enc_outputs)  # dec_logits : [batch_size x src_vocab_size x tgt_vocab_size]

        return dec_logits




