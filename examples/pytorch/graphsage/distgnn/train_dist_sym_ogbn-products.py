"""
Inductive Representation Learning on Large Graphs
Paper: http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf
Code: https://github.com/williamleif/graphsage-simple
Simple reference implementation of GraphSAGE.
"""
import os
import sys
import psutil
import json
import argparse
import time
import numpy as np
import networkx as nx
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data.utils import load_tensors, load_graphs
from dgl.data import register_data_args, load_data
from dgl.nn.pytorch.conv import SAGEConv

import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from load_graph import load_reddit, load_ogb

from dgl.distgnn.drpa_sym_exact_cdx import drpa

from dist_bn1d_v1 import dist_batch_norm
from dgl.sparse import mfg, hopped_mfg
from preprocess_partitions import preprocess

try:
    import torch_ccl
except ImportError as e:
    print(e)

    
class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 aggregator_type,
                 leaf_node,
                 root_nid):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        if args.bn:
            self.bns = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        self.layers.append(SAGEConv(in_feats, n_hidden, aggregator_type))        
        if args.bn:
            self.bns.append(dist_batch_norm(n_hidden, root_nid, args.rank))

        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(SAGEConv(n_hidden, n_hidden, aggregator_type))
            if args.bn:
                self.bns.append(dist_batch_norm(n_hidden, root_nid, args.rank))
        # output layer
        self.layers.append(SAGEConv(n_hidden, n_classes, aggregator_type))


    ## Graphsage default class
    def forward(self, graph, inputs, flag=True):
        h = inputs
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                if args.bn:
                    tic = time.time()
                    h = self.bns[l](h)
                    if args.rank == 0: print("bn time: ", time.time() - tic)                        
                h = self.activation(h)
                h = self.dropout(h)
                
        return h
    
def evaluate(model, graph, features, labels, nid):
    model.eval()
    with th.no_grad():
        logits = model(graph, features, False)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels), correct.item(), len(labels)

    
def evaluate_dist(model, graph, features, labels, nid):
    model.eval()
    with th.no_grad():
        logits = model(graph, features, False)
        logits = logits[nid]
        labels = labels[nid]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item(), len(labels)
        
    
def run(g, data):
    n_classes, node_map, num_parts, rank, world_size, leaf_node, root_nid = data
    
    features = g.ndata['feat']
    labels = g.ndata['label']
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    test_mask = g.ndata['test_mask']
    in_feats = features.shape[1]
    n_edges = g.number_of_edges()
    #print("""----Data statistics------'
    ##Nodes %d
    #  #Edges %d
    #  #Classes %d
    #  #Train samples %d
    #  #Val samples %d
    #  #Test samples %d""" %
    #      (g.number_of_nodes(), n_edges, n_classes,
    #       train_mask.int().sum().item(),
    #       val_mask.int().sum().item(),
    #       test_mask.int().sum().item()))

    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        th.cuda.set_device(args.gpu)
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()

        test_mask = test_mask.cuda()
        print("use cuda:", args.gpu)

    train_nid = th.nonzero(train_mask, as_tuple=True)[0]
    val_nid = th.nonzero(val_mask, as_tuple=True)[0]
    test_nid = th.nonzero(test_mask, as_tuple=True)[0]        

    # graph preprocess and calculate normalization factor
    n_edges = g.number_of_edges()
    if cuda:
        g = g.int().to(args.gpu)

    th.manual_seed(42)    
    # create GraphSAGE model
    model = GraphSAGE(in_feats,
                      args.n_hidden,
                      n_classes,
                      args.n_layers,
                      F.relu,
                      args.dropout,
                      args.aggregator_type,
                      leaf_node,
                      root_nid)

    if args.rank == 0:
        print("Distributed data parallel model...")
        print("lr rate: ", args.lr)
        
    model = th.nn.parallel.DistributedDataParallel(model)
    if cuda:
        model.cuda()

    # use optimizer
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    gr, _ = load_ogb(args.dataset)
    gr_features = gr.ndata['feat']
    gr_labels = gr.ndata['labels']
    gr_train_mask = gr.ndata['train_mask']
    gr_train_nid = th.nonzero(gr_train_mask, as_tuple=True)[0]
    gr_test_mask = gr.ndata['test_mask']
    gr_test_nid = th.nonzero(gr_test_mask, as_tuple=True)[0]    
    
    gr_val_mask = gr.ndata['val_mask']
    gr_val_nid = th.nonzero(gr_val_mask, as_tuple=True)[0]
    
    # initialize graph
    dbest_acc = 0    
    best_acc = 0
    dur = []
    for epoch in range(args.n_epochs):
        tic = time.time()
        model.train()
        
        # forward
        logits = model(g, features)
        loss = F.cross_entropy(logits[train_nid], labels[train_nid])
        optimizer.zero_grad()
        loss.backward()
        
        for param in model.parameters():
            if param.requires_grad and param.grad is not None:
                th.distributed.all_reduce(param.grad.data,
                                          op=th.distributed.ReduceOp.SUM)

        
        optimizer.step()
        toc = time.time()
        if args.val:
            acc_train, nr, dr = evaluate(model, gr, gr_features, gr_labels, gr_train_nid)
            acc_val, nr, dr = evaluate(model, gr, gr_features, gr_labels, gr_val_nid)
            if rank == 0:
                print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy, train {:.4f}, val {:.4f}"
                      .format(epoch, toc - tic, loss.item(),
                              acc_train, acc_val), flush=True)
            
                                
        if args.rank == 0:
            print("Epoch: {} time: {:0.4} sec"
                  .format(epoch, toc - tic), flush=True)
            print()
            
        
        #if args.rank == 0 and (epoch %5 == 0 or epoch == args.n_epochs - 1):
        if (epoch %5 == 0 or epoch == args.n_epochs - 1):
            acc, nr, dr = evaluate(model, gr, gr_features, gr_labels, gr_test_nid)
            if args.rank == 0:
                if best_acc < acc: best_acc = acc
                print("#############################################################", flush=True)
                print("test size: ", gr_test_nid.size())
                print("Single node accuracy: Avg: {:0.4f}, best_acc: {:.4f}".
                      format(acc, best_acc), flush=True)
                print("#############################################################", flush=True)
            
    
def main(graph, data, dataset):
    run(graph, data)

class args_:
    def __init__(self, dataset):
        self.dataset = dataset
        print("dataset set to: ", self.dataset)


if __name__ == '__main__':
    th.set_printoptions(threshold=10)
    
    parser = argparse.ArgumentParser(description='GraphSAGE')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--nr", type=int, default=0,
                        help="#rounds in delayed updates")
    parser.add_argument("--dl", type=int, default=0,
                        help="#delay in delayed updates")        
    parser.add_argument("--val", default=False,
                        action='store_true')
    parser.add_argument("--bn", default=False,
                        action='store_true')    
    parser.add_argument("--lr", type=float, default= 0.03,
                        help="learning rate")
    parser.add_argument("--n-epochs", type=int, default=300,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--aggregator-type", type=str, default="gcn",
                        help="Aggregator type: mean/gcn/pool/lstm")
    parser.add_argument('--world-size', default=-1, type=int,
                         help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                         help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                         help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='ccl', type=str,
                            help='distributed backend')
    
    args = parser.parse_args()
    if args.rank == 0:
        print(args)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("PMI_SIZE", -1))
        if args.world_size == -1: args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1
    if args.distributed:
        args.rank = int(os.environ.get("PMI_RANK", -1))
        if args.rank == -1: args.rank = int(os.environ["RANK"])        
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("Rank: ", args.rank ," World_size: ", args.world_size)
    nc = args.world_size

    part_config = "./"
    if args.dataset == 'ogbn-products':
        part_config = os.path.join(part_config, "Libra_result_ogbn-products",\
                                   str(nc) + "Communities", "ogbn-products.json")
    else:
        print("Error: Dataset not found !!!")
        sys.exit(1)

    with open(part_config) as conf_f:
        part_metadata = json.load(conf_f)
            
        
    part_files = part_metadata['part-{}'.format(args.rank)]
    assert 'node_feats' in part_files, "the partition does not contain node features."
    assert 'edge_feats' in part_files, "the partition does not contain edge feature."
    assert 'part_graph' in part_files, "the partition does not contain graph structure."
    node_feats = load_tensors(part_files['node_feats'])
    # edge_feats = load_tensors(part_files['edge_feats'])
    graph = load_graphs(part_files['part_graph'])[0][0]

    num_parts = part_metadata['num_parts']
    node_map  = part_metadata['node_map']

    graph.ndata['feat'] = node_feats['feat']
    graph.ndata['label'] = node_feats['label']
    graph.ndata['train_mask'] = node_feats['train_mask']
    graph.ndata['test_mask'] = node_feats['test_mask']
    graph.ndata['val_mask'] = node_feats['val_mask']

    if args.dataset == 'ogbn-products':
        print("Loading ogbn-products")
        g_orig, _ = load_ogb('ogbn-products')
    elif args.dataset == 'ogbn-papers100M':
        print("Loading ogbn-papers100M")
        g_orig, _ = load_ogb('ogbn-papers100M')        
    else:
        g_orig = load_data(args)[0]

    try:
        labels = g_orig.ndata['labels'][np.arange(g_orig.number_of_nodes())]
    except:
        labels = g_orig.ndata['label'][np.arange(g_orig.number_of_nodes())]
        
    n_classes = len(th.unique(labels[th.logical_not(th.isnan(labels))]))
    if args.rank == 0:
        print("n_classes: ", n_classes, flush=True)

    leaf_node, root_nid = preprocess(graph, node_map, args.rank)
    data = n_classes, node_map, num_parts, args.rank, args.world_size, leaf_node, root_nid
    gobj = drpa(graph, args.rank, num_parts, node_map, args.nr, args.dl, dist, args.n_layers, \
                leaf_node)

    main(gobj, data, args.dataset)
    
    if args.rank == 0:
        print("run details:")
        gobj.printv()
        print("#ranks: ", args.world_size)
        print("Dataset:", args.dataset)
        print("Delay: ", args.nr)        
        print("lr: ", args.lr)
        print("bn: ", args.bn)
        print("nr: ", args.nr)
        print("dl: ", args.dl)
        print("backend: ", args.dist_backend)
        print("Aggregator: ", args.aggregator_type)
        print()
        print()

    gobj.drpa_finalize()        

