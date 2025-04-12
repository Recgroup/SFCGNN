import argparse
import models
from utils import train, val_and_test
from data import load_data
import torch
import  random
import os
import time
import numpy as np
from GT import *


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

#parameter
earStopStep = 100
parser = argparse.ArgumentParser() # citeseer 50 epochs
parser.add_argument('--data', type=str, default='cora', help='{cora, pubmed, citeseer}.')
parser.add_argument('--model', type=str, default='SFCGNN', help='')
parser.add_argument('--hid', type=int, default= 128, help='Number of hidden units.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--wightDecay', type=float, default=5e-4, help='Weight decay .')
parser.add_argument('--nlayer', type=int, default=16,  help='Number of layers, works for Deep model.')
parser.add_argument("--seed",type=int,default=30,help="seed for model")

parser.add_argument("--tau",type=float,default=0.5,help="tau for CL")
parser.add_argument('--dropout', type=float, default=0.4,help='Dropout rate.')
parser.add_argument('--maskingRate', type=float, default=0.3,help='dropout for CL')
parser.add_argument('--aug_adj', type=int, default=1,help='0 or 1') # 0 close aug adj ,1 open aug adj
parser.add_argument('--t', type=int, default=2,help='t={1,2,3,4,5}')

args = parser.parse_args()
set_seed(args.seed)

data = load_data(args.data)
nfeat = data.num_features
nclass = int(data.y.max())+1

all_test_acc = []
all_metric = []
start_time = time.time()
data.adj = [data.adj]

for i in range(1):
    stopStep = 0
    ###################################################train one stage
    ###################################################train one stage
    net = getattr(models, 'GCN')(args, nfeat, nclass)
    net = net.cuda() if torch.cuda.is_available() else net.cpu()
    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wightDecay)
    criterion = torch.nn.CrossEntropyLoss()
    best_val=0
    best_test=0

    for epoch in range(args.epochs):
        train_loss, train_acc ,_ ,_= train(net, args.model,optimizer, criterion, data)
        test_acc,val_acc,metric1,metric2 = val_and_test(net, args.model,data)

        if best_val < val_acc:
            best_val = val_acc
            torch.save(net.state_dict(), f'{args.data}_best_dgi.pkl')

        else:
            stopStep+=1

        if stopStep>earStopStep:
            break

        print("pr-train , epoch:{} train_acc:{},train_loss:{},val_acc:{}"
              .format(epoch,round(train_acc.tolist(),3),
             round(train_loss.tolist(),3),
             round(val_acc.tolist(),3)))

    ################################################### rebuild adj
    ################################################### rebuild adj
    net.load_state_dict(torch.load(f'{args.data}_best_dgi.pkl'))
    #get pred
    pre,_,_,_ = net(data.x, data.adj)
    # get aug adj by topk preds
    data.adj = generateAugAdj(pre,data.adj[0],1 ,args.t,args.aug_adj)

    ###################################################train two stage
    ###################################################train two stage
    net = getattr(models, args.model)(args, nfeat, nclass)
    net = net.cuda() if torch.cuda.is_available() else net.cpu()
    optimizer = torch.optim.Adam(net.parameters(), args.lr, weight_decay=args.wightDecay)
    criterion = torch.nn.CrossEntropyLoss()
    best_val = 0
    best_test = 0
    best_metric = [0, 0]

    for epoch in range(args.epochs):
        train_loss, train_acc, _, _ = train(net, args.model, optimizer, criterion, data)
        test_acc, val_acc, metric1, metric2 = val_and_test(net, args.model, data)

        if best_val < val_acc:
            best_val = val_acc
            best_test = test_acc
            best_metric = [metric1, metric2]
            stopStep=0
        else:
            stopStep+=1

        if stopStep>earStopStep:
            break
        print("epoch:{} train_acc:{},train_loss:{},val_acc:{},test_acc:{},best_acc:{}"
              .format(epoch, round(train_acc.tolist(), 3),
                      round(train_loss.tolist(), 3),
                      round(val_acc.tolist(), 3),
                      round(test_acc.tolist(), 3),
                      round(best_test.tolist(), 3)))


    print("\ncurrent best test acc",best_test)
    all_test_acc.append(best_test)
    all_metric.append(best_metric)
end_time = time.time()
