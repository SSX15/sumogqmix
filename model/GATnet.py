
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from env.env_MAL import att_list

class GraphAttentionLayer(nn.Module):

    # layer层初始化中，除去基本的输入输出层，还需要指定alpha,concat
    # alpha 用于指定激活函数LeakyRelu中的参数
    # concat用于指定该层输出是否要拼接，因为用到了多头注意力机制
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()

        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # W表示该层的特征变化矩阵
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        # 一种初始化的方法
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        # a表示用于计算注意力系数的单层前馈神经网络。
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj, eval=False):
        # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Wh = torch.matmul(h, self.W)

        # 用于得到计算注意力系数的矩阵。
        # 这里采用的是矩阵形式，一次计算便可得到网络中所有结点对之间的注意力系数
        a_input = self._prepare_attentional_mechanism_input(Wh)

        # a_input的形状为(n, n, 2*out_features)
        # a 的形状为(2*out_features, 1)
        # 二者相乘则为(n, n, 1)，需要通过 squeeze操作去掉第3个维度
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(4))

        # 注意力系数可能为0，这里需要进行筛选操作，便于后续乘法
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=3)
        attention = F.dropout(attention, self.dropout, training=self.training)

        if eval:
            att_list.append(attention.view(-1).tolist())
        # 输出结合注意力系数的特征矩阵
        h_prime = torch.matmul(attention, Wh)

        # 如果是多头拼接，则进行激活，反之不必
        if self.concat:
            return F.elu(h_prime)

        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):

        # number of nodes
        N = Wh.size()[2]

        # 这里为了通过一次矩阵相乘来得到所有结点对之间的注意力系数
        # 需要构造一种特殊的矩阵，该矩阵的形式为
        # h1,h1
        # h1,h2
        # .
        # .
        # .
        # h1,hn
        # h2,h1
        # .
        # .
        # hn,hn
        # 该矩阵的形状为(n*n, 2*out_features)

        # 用于生成(h1,...,h1,h2,...,h2,...,hn)^T
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=2)
        # 用于生成(h1,h2,...,hn,h1,...hn)^T
        Wh_repeated_alternating = Wh.repeat(1, 1, N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=3)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        # 将维度改为(N, N, 2 * self.out_features)
        return all_combinations_matrix.view(Wh.shape[0], Wh.shape[1], N, N, 2 * self.out_features)

class GAT(nn.Module):

    # GAT实现了多头注意力，这里需要指定头数nheads
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        super(GAT, self).__init__()
        self.dropout = dropout
        # 注意力部分
        # 这里用生成式来表示多头注意力
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                           range(nheads)]

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        # 模型输出部分
        # GAT的输出层
        self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    # GAT的计算方式
    def forward(self, x, adj, eval=False):
        x = F.dropout(x, self.dropout, training=self.training)

        # 计算并拼接由多头注意力所产生的特征矩阵
        x = torch.cat([att(x, adj, eval) for att in self.attentions], dim=3)
        x = F.dropout(x, self.dropout, training=self.training)
        # 特征矩阵经由输出层得到最终的模型输出
        x = F.elu(self.out_att(x, adj))
        return F.log_softmax(x, dim=1)


class GATNet(nn.Module):
    def __init__(self, args):
        super(GATNet, self).__init__()
        self.args = args
        self.adj = args.adj.to(self.args.device)
        self.input_dim = self.args.tl_ob_dim
        self.gat_dim = self.args.gat_dim
        self.heads = 1
        self.eb = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.ReLU()
        )

        self.gat = GAT(nfeat=self.gat_dim, nhid=self.gat_dim, nclass=self.gat_dim, dropout=0, alpha=1, nheads=self.heads)



    def forward(self, x, eval=False): #x:(batch, seq_len, n, feature)
        x = self.eb(x)
        x = self.gat(x, self.adj, eval)
        return x



