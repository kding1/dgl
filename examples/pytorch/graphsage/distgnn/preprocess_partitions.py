import torch as th

def node2part(lf, node_map):
    pos = 0
    for nm in node_map:
        if lf < int(nm):
            return pos
        pos += 1

def preprocess(graph, node_map, rank):
    if rank == 0:
        print("Preprocessing...")
        
    lftensor = graph.ndata['lf']
    leaf_node = th.ones(lftensor.shape[0], dtype=th.int32)
    
    nsn = 0
    snp = 0
    for i in range(lftensor.shape[0]):
        lf = int(lftensor[i])
        nsn += 1
        if lf != -200:
            p = node2part(lf, node_map)
            if p != rank:
                snp += 1
                graph.ndata['train_mask'][i] = 0
                graph.ndata['test_mask'][i] = 0
                graph.ndata['val_mask'][i] = 0                
                leaf_node[i] = 0

    train_mask = graph.ndata['train_mask']
    train_nid = th.nonzero(train_mask, as_tuple=True)[0]    
    #print("After rooting train size: ", train_nid.size(), flush=True)
    root_nid = th.nonzero(leaf_node, as_tuple=True)[0]

    test_mask = graph.ndata['test_mask']
    test_nid = th.nonzero(test_mask, as_tuple=True)[0]        
    train_size_t = th.tensor(int(train_nid.shape[0]), dtype=th.int64)
    th.distributed.all_reduce(train_size_t, op=th.distributed.ReduceOp.SUM)
    test_size_t = th.tensor(int(test_nid.shape[0]), dtype=th.int64)
    th.distributed.all_reduce(test_size_t, op=th.distributed.ReduceOp.SUM)
    
    #if args.rank == 0:
    #    print("##### original training nodes: {}, test nodes: {}".format(
    #          int(train_size_t), int(test_size_t)))
    if rank == 0:
        print("Preprocessing done.")
        
    return leaf_node, root_nid
