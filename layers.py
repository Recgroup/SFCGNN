import torch
import torch.nn as nn
from  data import  *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math



class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class LinearAttention(nn.Module):
    def __init__(self, args,hidden_size):
        super(LinearAttention, self).__init__()
        self.args = args
        self.hid = hidden_size
        self.att = nn.Sequential(nn.Linear(2*self.hid,1))

    def forward(self, Q, K):   # h 2708XcX64
        Q = Q.repeat(1, K.shape[1], 1)
        # attention = torch.softmax(math.sqrt(self.hid)*F.dropout(self.att(torch.cat([Q, K], dim=-1)).squeeze(-1),0.2), dim=-1)
        attention = torch.softmax(self.att(torch.cat([Q, K], dim=-1)).squeeze(-1), dim=-1)
        return attention


class Local_Layer(nn.Module):
    def __init__(self, args,hid,head_num,nclass):
        super(Local_Layer, self).__init__()

        self.hid = hid
        self.nclass = nclass
        self.head_dim = hid // head_num
        self.head_num = head_num
        self.attentionLayers = nn.ModuleList([LinearAttention(args,self.head_dim) for _ in range(self.head_num)])

    def  forward(self,h,s):
        s = s.view(s.shape[0],-1,self.head_num,self.head_dim)
        h = h.view(h.shape[0],-1,self.head_num,self.head_dim) #.expand_as(s)
        attentions = [attentionLayer(h[:,:,idx,:],s[:,:,idx,:]).unsqueeze(1) for idx,attentionLayer  in enumerate(self.attentionLayers)]
        h = torch.cat([torch.matmul(attention,s[:,:,idx,:]) for idx,attention in enumerate(attentions)],dim=-1)
        return h.squeeze(1)
