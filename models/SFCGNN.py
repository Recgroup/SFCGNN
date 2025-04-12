import torch
from layers import *
import torch.nn as nn
import torch.nn.functional as F
from GT import *


class GCN(nn.Module):
    def __init__(self,args, nfeat, nclass):
        super(GCN, self).__init__()
        nhid=args.hid
        self.tau = args.tau
        self.maskingRate = args.maskingRate
        dropout = 0.6
        self.gc1 = GCNLayer(nfeat, nhid)
        self.gc2 = GCNLayer(nhid, nclass)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(True)

    def forward(self, x, adj):
        h = self.dropout(x)
        h = self.gc1(h, adj[0])
        h = adj[0].mm(h)

        h = self.relu(h)
        h = self.dropout(h)
        h = self.gc2(h, adj[0])
        h = adj[0].mm(h)
        return h, 0,0,0

class SFCGNN(nn.Module):
    def __init__(self, args, nfeat, nclass):
        super(SFCGNN, self).__init__()

        self.args =args
        self.nclass = nclass
        self.hid  = args.hid
        self.layers = args.nlayer
        self.tau = args.tau
        self.head_num = 1
        self.dropoutRate = args.dropout
        self.maskingRate = args.maskingRate
        self.local =  True if args.local >0 else False
        self.mergeWay = args.mergeWay

        #drop
        self.dropout = nn.Dropout(p=self.dropoutRate).train()
        self.fc = torch.nn.Linear(nfeat,self.hid)
        self.classify = torch.nn.Linear(self.hid,self.nclass)

        if self.local: #local
            self.local_Layer = Local_Layer(self.args,self.hid,self.head_num,self.nclass)

    def masking(self, input, mask_prob=0.3):
        random_mask = torch.empty(
            (input.shape[0], input.shape[1]),
            dtype=torch.float32,
            device=input.device).uniform_(0, 1) > mask_prob
        return random_mask * input

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def contrastiveloss(self, z1,z2,adj):
        z1 = self.masking(z1,self.maskingRate)
        z2 = self.masking(z2,self.maskingRate)
        f = lambda x: torch.exp(x / self.tau)
        adj_mask=torch.where(adj.to_dense().cpu() > 0, torch.ones(1), torch.zeros(1)).to(adj.device)
        refl_si1 = f(self.sim(z1, adj.mm(z2)))
        refl_si2 = f(self.sim(adj.mm(z1), adj.mm(adj.mm(z2))))
        CT = -torch.log(((refl_si1*adj_mask + refl_si2*adj_mask).sum(1)) / (refl_si1.sum(1)-(refl_si1*adj_mask + refl_si2*adj_mask).sum(1)
                                                               +refl_si2.sum(1) ))
        return CT.mean()

    def getEmbedding(self,input, adj):
        adj_ori =adj[0]
        adj_aug = adj[1]
        loss = 0
        h = self.fc(self.dropout(input))
        loss += self.layers * self.contrastiveloss(h, h, adj_aug)

        # #compute
        h0 = h
        s =  h

        for idx in range(self.layers):
            if self.local:
                neighborinfo = adj_ori.mm(self.dropout(h))
                s = torch.cat([neighborinfo, s], dim=-1) #if self.mergeWay==1 else self.merge(s,neighborinfo)
                h = self.local_Layer(h0, s)
            else:
                h = adj_ori.mm(h)
        return h,loss

    def getPre(self,h):
        h = self.dropout(h)
        y = self.classify(h) #torch.softmax(self.classify(h), dim=-1) if self.args.data in ['citeseer', 'cora', 'pubmed'] else self.classify(h)
        return y

    def forward(self, input, adj):
        #get embedding
        h,loss = self.getEmbedding(input, adj)
        # classifing
        y = self.getPre(h)
        return y,loss,0,0
