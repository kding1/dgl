import torch
import time
from torch import nn
from torch.nn import functional as F
import os, psutil
import gc
import sys

from .. import function as fn
from ..utils import expand_as_pair, check_eq_shape
from .. import DGLHeteroGraph
from math import ceil, floor

## fwdpass
from ..sparse import fdrpa_scatter_reduce_v41, fdrpa_gather_emb_v41, fdrpa_scatter_reduce_v42, fdrpa_gather_emb_v42, fdrpa_get_buckets_v4, deg_div, deg_div_back, fdrpa_init_buckets_v4

##backpass
from ..sparse import bdrpa_scatter_reduce_v41, bdrpa_gather_emb_v41, bdrpa_scatter_reduce_v42, bdrpa_gather_emb_v42, bdrpa_grad_normalize, rootify2

from ..sparse import bdrpa_scatter_reduce_v51, bdrpa_gather_emb_v51, bdrpa_scatter_reduce_v52, bdrpa_gather_emb_v52
from ..sparse import bdrpa_get_buckets_v6, bdrpa_gather_emb_v61, bdrpa_scatter_reduce_v61, bdrpa_scatter_reduce_v62, drpa_comm_iters


display = True
class gqueue():
    def __init__(self):
        self.queue = []

    def push(self, val):
        self.queue.append(val)

    def pop(self):
        if self.empty():
            return -1
        return self.queue.pop(0)

    def size(self):
        return len(self.queue)

    def printq(self):
        print(self.queue)

    def empty(self):
        if (len(self.queue)) == 0:
            return True
        else:
            return False

    def purge(self):
        self.queue = []

gfqueue  = gqueue()
gfqueue2 = gqueue()
gfqueue_fp2 = gqueue()
gbqueue  = gqueue()
gbqueue2 = gqueue()
gbqueue_fp2 = gqueue()

## sym
fstorage_comm_feats_async  = gqueue()
fstorage_comm_feats_async2  = gqueue()
fstorage_comm_feats_chunk_async  = gqueue()
fstorage_comm_feats_chunk_async2  = gqueue()
fstorage_comm_nodes_async  = gqueue()
fstorage_comm_nodes_async2 = gqueue()
fstorage_comm_nodes_async_fp2 = gqueue()
fstorage_comm_nodes_async_fp22 = gqueue()

fstorage_comm_iter = gqueue()
fp2_fstorage_comm_feats_async = gqueue()
fp2_fstorage_comm_feats_chunk_async = gqueue()
fp2_fstorage_comm_iter = gqueue()
fstorage_comm_nodes_async_fp22_ext = gqueue()

## backpass
bstorage_comm_feats_async  = gqueue()
bstorage_comm_feats_async2  = gqueue()
bstorage_comm_feats_chunk_async  = gqueue()
bstorage_comm_feats_chunk_async2  = gqueue()
bstorage_comm_nodes_async  = gqueue()
bstorage_comm_nodes_async2 = gqueue()
bstorage_comm_nodes_async_fp2 = gqueue()
bstorage_comm_nodes_async_fp22 = gqueue()

bstorage_comm_iter = gqueue()
bstorage_comm_iter2 = gqueue()
fp2_bstorage_comm_feats_async = gqueue()
fp2_bstorage_comm_feats_chunk_async = gqueue()
fp2_bstorage_comm_iter = gqueue()
bstorage_comm_nodes_async_fp22_ext = gqueue()
bstorage_recv_list_nodes = gqueue()
bstorage_out_size_nodes = gqueue()


class bpreset():
    def __init__(self, adj, selected_nodes, node_map_t, lftensor, num_parts, dist, rank):
        #self.nrounds = nrounds
        width = adj.shape[1]
        self.num_parts = num_parts
        self.buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        self.num_sel_nodes = torch.zeros(1, dtype=torch.int32)
        self.sel_nodes = torch.empty(selected_nodes.shape[0] * (num_parts - 1), dtype=torch.int32)
        self.node2part = torch.empty(selected_nodes.shape[0] * (num_parts - 1), dtype=torch.int32)
        self.node2part_index = torch.empty(selected_nodes.shape[0] * (num_parts - 1), dtype=torch.int32)
        bdrpa_get_buckets_v6(adj, selected_nodes, self.sel_nodes, self.node2part,
                             self.node2part_index, node_map_t, self.buckets,
                             lftensor, self.num_sel_nodes, width, num_parts, rank)
        self.input_sr = []
        for i in range(0, self.num_parts):
            self.input_sr.append(torch.tensor([self.buckets[i]], dtype=torch.int64))

        self.output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, self.num_parts)]
        #sync_req = self.dist.all_to_all(self.output_sr, self.input_sr, async_op=True)   # make it async
        sync_req = dist.all_to_all(self.output_sr, self.input_sr, async_op=True)   # make it async
        sync_req.wait() ## recv the #nodes communicated

        self.input_sr_t = torch.tensor(self.input_sr, dtype=torch.int64)
        self.output_sr_t = torch.tensor(self.output_sr, dtype=torch.int64)


class bpreset_v2():
    def __init__(self, adj, selected_nodes, node_map_t, lftensor, num_parts, dist, rank):
        #self.nrounds = nrounds
        width = adj.shape[1]
        self.selected_nodes = selected_nodes
        self.num_parts = num_parts
        self.buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        self.node2part = torch.empty(selected_nodes.shape[0], dtype=torch.int32)
        self.node2part_index = torch.empty(selected_nodes.shape[0], dtype=torch.int32)
        fdrpa_get_buckets_v4(adj, selected_nodes, self.node2part, self.node2part_index,
                             node_map_t, self.buckets, lftensor, width, num_parts, rank)
        
        self.input_sr = []
        for i in range(0, self.num_parts):
            self.input_sr.append(torch.tensor([self.buckets[i]], dtype=torch.int64))
        
        self.output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, self.num_parts)]
        # sync_req = self.dist.all_to_all(self.output_sr, self.input_sr, async_op=True)
        sync_req = dist.all_to_all(self.output_sr, self.input_sr, async_op=True)
        sync_req.wait() ## recv the #nodes communicated

        self.input_sr_t = torch.tensor(self.input_sr, dtype=torch.int64)
        self.output_sr_t = torch.tensor(self.output_sr, dtype=torch.int64)
        
        
class fpreset():
    def __init__(self, adj, selected_nodes, node_map_t, lftensor, num_parts, dist, rank):
        #self.nrounds = nrounds
        width = adj.shape[1]
        self.selected_nodes = selected_nodes
        self.num_parts = num_parts
        self.buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        self.node2part = torch.empty(selected_nodes.shape[0], dtype=torch.int32)
        self.node2part_index = torch.empty(selected_nodes.shape[0], dtype=torch.int32)
        fdrpa_get_buckets_v4(adj, selected_nodes, self.node2part, self.node2part_index,
                             node_map_t, self.buckets, lftensor, width, num_parts, rank)
        
        self.input_sr = []
        for i in range(0, self.num_parts):
            self.input_sr.append(torch.tensor([self.buckets[i]], dtype=torch.int64))
        
        self.output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, self.num_parts)]
        # sync_req = self.dist.all_to_all(self.output_sr, self.input_sr, async_op=True)
        sync_req = dist.all_to_all(self.output_sr, self.input_sr, async_op=True)
        sync_req.wait() ## recv the #nodes communicated

        self.input_sr_t = torch.tensor(self.input_sr, dtype=torch.int64)
        self.output_sr_t = torch.tensor(self.output_sr, dtype=torch.int64)

        
bpre_aux = []
fpre_aux = []
backpass = 1 ## by default backpass is on
bpass_open = 0 ## bpass activates after this value.
fpass_open = 0 ## fpass activates after this value.

## DRPA
def drpa(gobj, rank, num_parts, node_map, nrounds, delay, dist, nlayers, root_node, drpa_backpass,
         factive, bactive, split_train_nodes, split_notrain_nodes, L):
    global backpass, fpass_open, bpass_open
    backpass = drpa_backpass
    fpass_open = factive
    bpass_open = bactive
    if rank == 0:
        print("backpass: {}, fpass_open: {}, bpass_open: {}".format(backpass, factive, bactive))
        
    #d = drpa_master(gobj, rank, num_parts, node_map, nrounds, dist, nlayers)
    d = drpa_master(gobj._graph, gobj._ntypes, gobj._etypes, gobj._node_frames, gobj._edge_frames)
    d.drpa_init(rank, num_parts, node_map, nrounds, delay, dist, nlayers,
                root_node, split_train_nodes, split_notrain_nodes, L)
    return d

class drpa_master(DGLHeteroGraph):
    def drpa_init(self, rank, num_parts, node_map, nrounds, delay, dist, nlayers, root_node,
                  split_train_nodes, split_notrain_nodes, L):
        #print("In drpa Init....")
        self.rank = rank
        self.num_parts = num_parts
        self.node_map = node_map
        self.nrounds = nrounds
        self.delay = delay
        self.dist = dist
        self.nlayers = nlayers
        print("nlayers: ", nlayers)
        
        self.epochs_ar = [0 for i in range(self.nlayers)]
        self.epochi = 0
        self.gather_q41 = gqueue()
        self.output_sr_ar = []
        self.root_node = root_node
        self.split_train_nodes = split_train_nodes
        self.split_notrain_nodes = split_notrain_nodes
        self.L = L

        self.buf_neigh = 0
        if self.nrounds == -1: return
        ## Creates buckets based on ndrounds

        adj = self.dstdata['adj']
        lf = self.dstdata['lf']
        width = adj.shape[1]
        #print("width: ", width)
        ## groups send data according to 'send to rank'
        ## communicates and fill output_sr_ar for data to be recd.
        #print("Create bucket v1 activated!!!!!!!!!!!!!!!!")
        #self.drpa_create_buckets()
        #if rank == 0:
        #    print("Create bucket v2 activated!!!!!!!!!!!!!!!!")
        #self.drpa_create_buckets_v2() ## 100% split node comms
        #if rank == 0:
            #print("Create bucket v3 activated!!!!!!!!!!!!!!!!")
        #self.drpa_create_buckets_v3() ## 100% train-split train node w/ bucketing option in notrain nodes
        if rank == 0:
            print("Create bucket v4 activated!!!!!!!!!!!!!!!!")
        self.drpa_create_buckets_v4() ## V2 w/layering in selected nodes, aimed at buffering L0

        self.drpa_init_buckets(adj, lf, width)
        

    def drpa_finalize(self):
        if self.rank == 0:
            print("Symm: Clearning backlogs", flush=True)
        while gfqueue.empty() == False:
            req = gfqueue.pop()
            req.wait()
            req = gfqueue2.pop()
            req.wait()
        while gfqueue_fp2.empty() == False:
            req = gfqueue_fp2.pop()
            req.wait()

        fstorage_comm_feats_async.purge()
        fstorage_comm_feats_async2.purge()
        fstorage_comm_feats_chunk_async.purge()
        fstorage_comm_feats_chunk_async2.purge()
        fstorage_comm_nodes_async_fp2.purge()
        fstorage_comm_nodes_async_fp22.purge()
        fstorage_comm_iter.purge()
        fp2_fstorage_comm_feats_async.purge()
        fp2_fstorage_comm_feats_chunk_async.purge()
        fp2_fstorage_comm_iter.purge()

        while gbqueue.empty() == False:
            req = gbqueue.pop()
            req.wait()
            req = gbqueue2.pop()
            req.wait()
        while gbqueue_fp2.empty() == False:
            req = gbqueue_fp2.pop()
            req.wait()

        bstorage_comm_feats_async.purge()
        bstorage_comm_feats_async2.purge()
        bstorage_comm_feats_chunk_async.purge()
        bstorage_comm_feats_chunk_async2.purge()
        bstorage_comm_nodes_async_fp2.purge()
        bstorage_comm_nodes_async_fp22.purge()
        bstorage_comm_iter.purge()
        fp2_bstorage_comm_feats_async.purge()
        fp2_bstorage_comm_feats_chunk_async.purge()
        fp2_bstorage_comm_iter.purge()


        if gfqueue.empty() == False:
            print("gfqueue not empty after backlogs flushing", flush=True)
        if gfqueue2.empty() == False:
            print("gfqueue2 not empty after backlogs flushing", flush=True)
        if gfqueue_fp2.empty() == False:
            print("gfqueue_fp2 not empty after backlogs flushing", flush=True)

        if self.rank == 0:
            print("Clearning backlogs ends ", flush=True)

    def qpurge_and_check(self):
        bstorage_comm_iter2.purge()
        bstorage_comm_nodes_async_fp22.purge()
        bstorage_out_size_nodes.purge()
        bstorage_recv_list_nodes.purge()
        bstorage_comm_nodes_async_fp2.purge()
        assert gbqueue_fp2.size() == 0
        assert fp2_bstorage_comm_feats_async.size() == 0
        assert fp2_bstorage_comm_feats_chunk_async.size() == 0
        assert fp2_bstorage_comm_iter.size() == 0
        assert gbqueue.size() == 0
        assert gbqueue2.size() == 0
        assert bstorage_comm_feats_async.size() == 0
        assert bstorage_comm_feats_async2.size() == 0
        assert bstorage_comm_feats_chunk_async.size() == 0
        assert bstorage_comm_feats_chunk_async2.size() == 0
        assert bstorage_comm_nodes_async_fp2.size() == 0
        assert bstorage_comm_nodes_async_fp22.size() == 0
        assert bstorage_comm_iter.size() == 0
                    

    def printv(self):
        print("drpa file name: ", os.path.basename(__file__), flush=True)
        
    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        assert self.rank != -1, "drpa not initialized !!!"

        self.qpurge_and_check()
        
        epoch = self.epochs_ar[self.epochi]
        neigh = self.dstdata['h']
        adj = self.dstdata['adj']
        inner_node = self.dstdata['inner_node']
        lftensor = self.dstdata['lf']
        feat_dst = self.dstdata['h']
        #self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)

        mean = 0
        if reduce_func.name == "mean":
            reduce_func = fn.sum('m', 'neigh')
            mean = 1

        disable_caching = True   ## first level agg caching
        if disable_caching or (epoch == 0 or (epoch > 0 and self.epochi > 0)):  ## pcl
            self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
            if self.nrounds != -1:
                #print("> Running w/ nrounds: ", self.nrounds)
                self.dstdata['h'] = call_drpa_core_FN_BLR(neigh, adj, inner_node,
                                                          lftensor, self.selected_nodes,
                                                          self.node_map, self.num_parts,
                                                          self.rank, epoch,
                                                          self.dist,
                                                          self.r_in_degs,
                                                          self.delay,
                                                          self.nrounds,
                                                          self.output_sr_ar,
                                                          self.gather_q41,
                                                          self.epochi,
                                                          self.nlayers,
                                                          self.root_node)
                self.srcdata['h'] = self.dstdata['h']
            
            #print("rfunc: ", reduce_func.name)
            #mean = 0
            #if reduce_func.name == "mean":
            #    reduce_func = fn.sum('m', 'neigh')
            #    mean = 1
            
            tic = time.time()
            DGLHeteroGraph.update_all(self, message_func, reduce_func)
            toc = time.time()
            
            if self.rank == 0  and display:
                print("Time for local aggregate: {:0.4f}, nrounds {}".format(toc - tic, self.nrounds))
            
            if self.nrounds == -1:
                if mean == 1:
                    feat_dst = self.dstdata['h']
                    self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
                    #self.dstdata['neigh'] = self.dstdata['neigh'] / (self.r_in_degs.unsqueeze(-1) + 1)
                    self.dstdata['neigh'] = self.dstdata['neigh'] / (self.r_in_degs.unsqueeze(-1))
                return
            
            neigh = self.dstdata['neigh']
            #adj = self.dstdata['adj']
            #inner_node = self.dstdata['inner_node']
            #lftensor = self.dstdata['lf']
            #feat_dst = self.dstdata['h']
            #epoch = self.epochs_ar[self.epochi]
            
            tic = time.time()
            # print(">> Running w/ nrounds: ", self.nrounds)
            self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
            self.dstdata['neigh'] = call_drpa_core_FF_BRL(neigh, adj, inner_node,
                                                          lftensor, self.selected_nodes,
                                                          self.selected_nodes_bck,
                                                          self.node_map, self.num_parts,
                                                          self.rank, epoch,
                                                          self.dist,
                                                          self.r_in_degs,
                                                          self.delay,
                                                          self.nrounds,
                                                          self.output_sr_ar,
                                                          self.gather_q41,
                                                          self.epochi,
                                                          self.nlayers,
                                                          1)
            toc = time.time()
            if self.rank == 0 and display:
                print("Time for remote aggregate: {:0.4f}".format(toc - tic))
                
            if mean == 1:
                #self.dstdata['neigh'] = self.dstdata['neigh'] / (self.r_in_degs.unsqueeze(-1) + 1)
                self.dstdata['neigh'] = self.dstdata['neigh'] / (self.r_in_degs.unsqueeze(-1))

        else:  ## pcl
            if self.rank == 0:
                print("Restoring layer-0 Aggregations..")
            self.dstdata['neigh'] = self.buf_neigh.detach().clone()

        if (not disable_caching) and (epoch == 0 and self.epochi == 0): ##pcl
            self.buf_neigh = self.dstdata['neigh'].detach().clone()
        #######################
            
        self.epochs_ar[self.epochi] += 1
        self.epochi = (self.epochi + 1) % (self.nlayers)

        
    def drpa_init_buckets(self, adj, lf, width):
        node_map_t = torch.tensor(self.node_map, dtype=torch.int32)
        if self.nrounds == 0:
            nrounds_ = 1
        else:
            nrounds_ = self.nrounds
        #for l in range(nrounds_):
        for l in range(self.nlayers):
            selected_nodes_t = torch.tensor(self.selected_nodes[l], dtype=torch.int32)
            faux = fpreset(adj, selected_nodes_t, node_map_t, lf, self.num_parts, self.dist, self.rank)
            fpre_aux.append(faux)

            selected_nodes_t = torch.tensor(self.selected_nodes[l-1], dtype=torch.int32)
            baux = bpreset_v2(adj, selected_nodes_t, node_map_t, lf, self.num_parts,
                              self.dist, self.rank)
            bpre_aux.append(baux)            

            
    def drpa_create_buckets(self):
        inner_nodex = self.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)

        if self.nrounds == 0:
            nrounds_ = 1
        else:
            nrounds_ = self.nrounds
        self.selected_nodes = [ [] for i in range(nrounds_)]  ## #native nodes

        # randomly divide the nodes in 5 rounds for comms
        total_alien_nodes = inner_nodex.count(0)  ## count split nodes
        alien_nodes_per_round = int((total_alien_nodes + nrounds_ -1) / nrounds_)

        counter = 0
        pos = 0
        r = 0
        while counter < n:
            if inner_nodex[counter] == 0:    ##split node
                self.selected_nodes[r].append(counter)
                pos += 1
                if pos % alien_nodes_per_round == 0:
                    r = r + 1

            counter += 1

        assert pos == n - idx
        if (counter != len(inner_nodex)):
            print("counter: ", counter, " ", len(inner_nodex))

        assert counter == len(inner_nodex), "assertion"
        #if pos == total_alien_nodes:
        #    print("pos: ", pos, " ", total_alien_nodes)
        assert pos == total_alien_nodes, "pos alien not matching!!"

        if self.rank == 0:
            print("Selected nodes in each round: ", flush=True)
            for i in range(nrounds_):
                print("round: ", i,  " nodes: ", len(self.selected_nodes[i]), flush=True);


    def drpa_create_buckets_v2(self):
        inner_nodex = self.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)

        if self.nrounds == 0:
            nrounds_ = 1
        else:        
            nrounds_ = self.nrounds
            
        self.selected_nodes = [ [] for i in range(nrounds_)]  ## #native nodes

        # randomly divide the nodes in 5 rounds for comms
        total_alien_nodes = inner_nodex.count(0)  ## count split nodes
        alien_nodes_per_round = int((total_alien_nodes + nrounds_ -1) / nrounds_)

        counter = 0
        pos = 0
        r = 0
        while counter < n:
            if inner_nodex[counter] == 0:    ##split node
                self.selected_nodes[r].append(counter)
                pos += 1
                #if pos % alien_nodes_per_round == 0:
                #    r = r + 1

            counter += 1

        assert pos == n - idx
        if (counter != len(inner_nodex)):
            print("counter: ", counter, " ", len(inner_nodex))

        assert counter == len(inner_nodex), "assertion"
        #if pos == total_alien_nodes:
        #    print("pos: ", pos, " ", total_alien_nodes)
        assert pos == total_alien_nodes, "pos alien not matching!!"

        for i in range(1, nrounds_):
            self.selected_nodes[i] = self.selected_nodes[0]
            
        if self.rank == 0:
            print("Selected nodes in each round: ", flush=True)
            for i in range(nrounds_):
                print("round: ", i,  " nodes: ", len(self.selected_nodes[i]), flush=True);

    
    ## for cd-0x each bucket contains all split training nodes and some partitions of non-train nodes
    def drpa_create_buckets_v3(self):
        inner_nodex = self.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)
        #assert n-idx == len(self.split_train_nodes) + len(self.split_notrain_nodes)
        #assert n-idx == self.split_train_nodes.shape[0] + self.split_notrain_nodes.shape[0]
        if self.nrounds == 0:
            nrounds_ = 1
        else:        
            nrounds_ = self.nrounds
            
        self.selected_nodes = [ [] for i in range(nrounds_)]  ## #native nodes

        # randomly divide the nodes in 5 rounds for comms
        total_alien_nodes = self.split_notrain_nodes.shape[0]  ## changed total split nodes here
        alien_nodes_per_round = int((total_alien_nodes + nrounds_ - 1) / nrounds_)
        #print("Non zero rounds: ", total_alien_nodes/alien_nodes_per_round)
        pos = 0
        r = 0
        for i in range(total_alien_nodes):
            nid = int(self.split_notrain_nodes[i])
            self.selected_nodes[r].append(nid)
            pos += 1
            if pos % alien_nodes_per_round == 0 and i!=total_alien_nodes - 1:
                r = r + 1

        if r!= nrounds_-1:
            print("{} {}".format(r, nrounds_ - 1))
        assert r == nrounds_-1
        #assert counter == len(inner_nodex), "assertion"
        #if pos == total_alien_nodes:
        #    print("pos: ", pos, " ", total_alien_nodes)
        assert pos == total_alien_nodes, "pos alien not matching!!"

        for i in range(nrounds_):
            self.selected_nodes[i] = self.selected_nodes[i] + self.split_train_nodes.tolist()
            
        if self.rank == 0:
            print("Total split nodes: {}, train split nodes: {} notrain split nodes: {},, n: {}".
                  format(n-idx, self.split_train_nodes.shape[0], self.split_notrain_nodes.shape[0], n))
            print("Selected nodes in each round: ", flush=True)
            for i in range(nrounds_):
                print("round: ", i,  " nodes: ", len(self.selected_nodes[i]), flush=True);


    ## assuming cd-0 w/ 100% split node comms --now w/ layered selected_nodes
    def drpa_create_buckets_v4(self):
        inner_nodex = self.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)

        nrounds_ = 1 if self.nrounds == 0 else self.nrounds
        #print("nrounds: ", nrounds_, flush=True)
        #self.selected_nodes = [ [[] for i in range(nrounds_)] for j in range(self.nlayers)]
        self.selected_nodes = [ [] for j in range(self.nlayers)]
        self.selected_nodes_bck = [ [] for j in range(self.nlayers)]
        #print("sel nodes: ", self.selected_nodes)
        # randomly divide the nodes in 5 rounds for comms
        total_alien_nodes = inner_nodex.count(0)  ## count split nodes
        alien_nodes_per_round = int((total_alien_nodes + nrounds_ -1) / nrounds_)

        #counter = 0
        #pos = 0
        #r = 0
        #while counter < n:
        #    if inner_nodex[counter] == 0:    ##split node
        #        for j in range(self.nlayers):
        #            print(j, " ", len(self.selected_nodes[j][r]))
        #            self.selected_nodes[j][r].append(counter)
        #        pos += 1
        #        #if pos % alien_nodes_per_round == 0:
        #        #    r = r + 1
        #
        #    counter += 1

        nid = torch.nonzero(~(self.ndata['inner_node'].bool()), as_tuple=True)[0]
        print(nid.size())
        pos = nid.shape[0]
        #tnid=[3,4,5]
        #nid=torch.tensor(tnid)        
        for j in range(self.nlayers):
            #for r in range(nrounds_):
            #print(j, " ", self.selected_nodes[j][r])
            #self.selected_nodes[j] = nid.tolist()
            self.selected_nodes[j] = self.L[j].tolist()
            #print((self.ndata['inner_node'][self.L[j].long()]).sum())
            #print(j, " ", len(self.selected_nodes[j][r]))

        #self.selected_nodes[0] = self.L[0].tolist()
        #self.selected_nodes[1] = self.L[0].tolist()
        #self.selected_nodes[2] = self.L[0].tolist()
        #self.selected_nodes[0] = nid.tolist()
        #self.selected_nodes[1] = nid.tolist()
        #self.selected_nodes[2] = nid.tolist()
        
        if self.rank == 0:
            print("Selected nodes in each round: ", flush=True)
            print("nlayers: ", self.nlayers, flush=True)
            print("nrounds: ", nrounds_, flush=True)
            for j in range(self.nlayers):
                #for i in range(nrounds_):
                print("L: ", j,  " nodes: ",
                      len(self.selected_nodes[j]), flush=True);

                
    def in_degrees(self):
        try:
            return self.r_in_degs
        except:
            print("Passing only local node degrees.")
            pass

        return DGLHeteroGraph.in_degrees(self)


def message(rank, msg, val=-1, val_=-1):
    if rank == 0 and display:
        if val == -1:
            print(msg, flush=True)
        elif val_ == -1:
            print(msg.format(val), flush=True)
        else:
            print(msg.format(val, val_), flush=True)


## 2nd drpa call
class drpa_core_FF_BRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, adj, inner_node, lftensor, selected_nodes, selected_nodes_bck,
                node_map, num_parts, rank, epoch, dist, in_degs, delay, nrounds,
                output_sr_ar, gather_q41, epochi, nlayers, pass_flag):

        #feat_copy = feat.clone()
        nrounds_update = nrounds
        if rank == 0 and display:
            print(" >>>>>>>>>>>> drpa core FF BRL fwd...{}".format(epochi))

        prof = []
        feat_size = feat.shape[1]
        int_threshold = pow(2, 31)/4 - 1              ##bytes
        base_chunk_size = int(int_threshold / num_parts) ##bytes
        base_chunk_size_fs = floor(base_chunk_size / (feat_size + 1) )
        roundn =  epoch % nrounds if nrounds > 0 else 0
        width = adj.shape[1]
        
        ## Here epochi is ith layer
        node_map_t = torch.tensor(node_map, dtype=torch.int32)
        selected_nodes_t = []

        nrounds_ = 1 if nrounds == 0 else nrounds
        #for sn in selected_nodes:
        for i in range(nrounds_):
            sn = selected_nodes[epochi]
            selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))
            #print("fwd 2 sel size: ", epochi, " ", selected_nodes_t[i].size())

        #if epoch >= 0*nrounds_update or nrounds == 0:
        #if epoch - fpass_open >= 0*nrounds_update:
        if epoch - fpass_open >= 0*delay:
            tic = time.time()
            #### 1. get bucket sizes
            """
            buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
            #node2part = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
            #node2part_index = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
            node2part = torch.empty(selected_nodes_t[roundn].shape[0], dtype=torch.int32)
            node2part_index = torch.empty(selected_nodes_t[roundn].shape[0], dtype=torch.int32)
            fdrpa_get_buckets_v4(adj, selected_nodes_t[roundn], node2part, node2part_index,
                                 node_map_t, buckets, lftensor, width, num_parts, rank)
            
            #message(rank, "Time for get buckets: {:0.4f}", (time.time() - tic))
            
            ###### comms to gather the bucket sizes for all-to-all feats
            input_sr = []
            for i in range(0, num_parts):
                input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))
            
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
            sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated
            """
            obj = fpre_aux[epochi]
            node2part = obj.node2part
            node2part_index = obj.node2part_index
            input_sr = obj.input_sr.copy()
            output_sr = obj.output_sr.copy()
            buckets = obj.buckets
            
            ### 3. gather emdeddings
            send_feat_len = 0
            in_size = []
            for i in range(num_parts):
                in_size.append(int(buckets[i]) * (feat_size + 1))
                send_feat_len += in_size[i]
                
            in_size_ = (buckets * (feat_size + 1)).tolist()
            send_feat_len_ = int(buckets.sum()) * (feat_size + 1)
            assert send_feat_len == send_feat_len_
            assert in_size == in_size_
            
            ## mpi call split starts
            ##############################################################################
            
            cum = 0; flg = 0
            for i in output_sr:
                cum += int(i) * (feat_size + 1)
                if int(i) >= base_chunk_size_fs: flg = 1
            
            for i in input_sr:
                flg = 1 if int(i) >= base_chunk_size_fs else None
            
            back_off = 1
            if cum >= int_threshold or send_feat_len >= int_threshold or flg:
                for i in range(num_parts):
                    val = ceil((int(input_sr[i]) ) / base_chunk_size_fs)
                    #if val > back_off:
                    #    back_off = val
                    back_off = val if val > back_off else back_off
            
            """
            obj = fpre_aux[roundn]
            back_off = drpa_comm_iters(obj.output_sr_t, obj.input_sr_t, feat_size,
                                       base_chunk_size_fs, num_parts, int(int_threshold))
            node2part = obj.node2part
            node2part_index = obj.node2part_index
            input_sr = obj.input_sr.copy()
            output_sr = obj.output_sr.copy()
            
            ## end of new code which had replaced the above commented code
            """
            tback_off = torch.tensor(back_off)
            rreq = torch.distributed.all_reduce(tback_off, op=torch.distributed.ReduceOp.MAX, async_op=True)
            
            tic = time.time()
            lim = 1
            soffset_base = [0 for i in range(num_parts)]  ## min chunk size
            soffset_cur = [0 for i in range(num_parts)]       ## send list of ints
            roffset_cur = [0 for i in range(num_parts)]       ## recv list of intrs
            
            j=0
            while j < lim:
                tsend = 0; trecv = 0
                for i in range(num_parts):
                    soffset_base[i] += soffset_cur[i]
                    if input_sr[i] < base_chunk_size_fs:
                        soffset_cur[i] = int(input_sr[i])
                        input_sr[i] = 0
                    else:
                        soffset_cur[i] = base_chunk_size_fs
                        input_sr[i] -= base_chunk_size_fs
            
                    if output_sr[i]  < base_chunk_size_fs:
                        roffset_cur[i] = int(output_sr[i])
                        output_sr[i] = 0
                    else:
                        roffset_cur[i] = base_chunk_size_fs
                        output_sr[i] -= base_chunk_size_fs
            
                    tsend += soffset_cur[i]
                    trecv += roffset_cur[i]
            
                send_node_list  = [torch.empty(soffset_cur[i], dtype=torch.int32) for i in range(num_parts)]
                sten_ = torch.empty(tsend * (feat_size + 1), dtype=feat.dtype)
                dten_ = torch.empty(trecv * (feat_size + 1), dtype=feat.dtype)
                sten_nodes_ = torch.empty(tsend , dtype=torch.int32)
                dten_nodes_ = torch.empty(trecv , dtype=torch.int32)
            
                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                out_size_nodes = [0 for i in range(num_parts)]
                in_size_nodes = [0 for i in range(num_parts)]
            
                offset = 0
                for i in range(num_parts): ## gather by followers
                    fdrpa_gather_emb_v41(feat, feat.shape[0], adj, sten_, offset,
                                         send_node_list[i], sten_nodes_,
                                         selected_nodes_t[roundn],
                                         in_degs, node2part, node2part_index, width, feat_size, i,
                                         soffset_base[i], soffset_cur[i], node_map_t, num_parts)
            
                    out_size[i]       = roffset_cur[i] * (feat_size + 1)
                    in_size[i]        = soffset_cur[i] * (feat_size + 1)
                    offset            += soffset_cur[i]
                    out_size_nodes[i] = roffset_cur[i]
                    in_size_nodes[i]  = soffset_cur[i]
            
                #message(rank, "Sending {}, recving {} data I", tsend, trecv)
                req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)
                gfqueue.push(req)
                req2 = dist.all_to_all_single(dten_nodes_, sten_nodes_,
                                             out_size_nodes, in_size_nodes,
                                             async_op=True)
                gfqueue2.push(req2)
                    
                soffset_cur_copy = soffset_cur.copy()
            
    	        ## section III: store pointers for the data in motion
                #print("count, lim: ", lim, " ", fstorage_comm_feats_async.size())
                fstorage_comm_feats_async.push(dten_)
                fstorage_comm_feats_async2.push(dten_nodes_)
                fstorage_comm_feats_chunk_async.push(out_size)
                fstorage_comm_feats_chunk_async2.push(out_size_nodes)
                fstorage_comm_nodes_async_fp2.push(send_node_list)    ## fwd phase II
                fstorage_comm_nodes_async_fp22.push(soffset_cur_copy) ## fwd phase II
            
                if j == 0:
                    rreq.wait()
                    lim = int(tback_off)
                j = j + 1
            ##############################################################################
            ## mpi call split ends
            fstorage_comm_iter.push(lim)

            #message(rank, "Max iters in MPI split comm: {}", (lim))
            prof.append('-Gather I: {:0.4f}'.format(time.time() - tic))

        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []
        #if epoch >= nrounds_update or nrounds == 0:
        #if epoch - fpass_open >= nrounds_update:
        if epoch - fpass_open >= delay:
            assert gfqueue.empty() == False, "Error: Forward empty queue !!!"

            ticg = time.time()
            lim = fstorage_comm_iter.pop()
            out_size_nodes_ar = []
            for i in range(lim):
                if rank == 0 and display: tic = time.time()

                req = gfqueue.pop();  req.wait()
                req = gfqueue2.pop(); req.wait()

                #message(rank, "Time for async comms I: {:4f}", (time.time() - tic))
                prof.append('Async comm I: {:0.4f}'.format(time.time() - tic))
                
                # assert fstorage_comm_feats_async.size() == nrounds*nlayers + 1
                #if fstorage_comm_feats_async.size() != nrounds*nlayers + 1:
                 #   print("Assertion failed: ", fstorage_comm_feats_async.size(), " lim: ", nlayers)
                    
                otf = fstorage_comm_feats_async.pop()
                out_size = fstorage_comm_feats_chunk_async.pop()
                otn = fstorage_comm_feats_async2.pop()
                out_size_nodes = fstorage_comm_feats_chunk_async2.pop()
                out_size_nodes_ar.append(out_size_nodes)
                
                recv_list_nodes_ar = []; ilen = 0
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))

                pos = torch.tensor([0], dtype=torch.int64)
                offsetf = 0; offsetn = 0
                for l in range(num_parts):
                    fdrpa_scatter_reduce_v41(otf, offsetf, otn, offsetn,
                                       feat, in_degs, node_map_t, out_size[l], feat_size,
                                       num_parts, recv_list_nodes_ar[l], pos,
                                       int(out_size_nodes[l]), rank)

                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]

                assert ilen == pos[0], "Issue in scatter reduce!"
                recv_list_nodes.append(recv_list_nodes_ar)

            #message(rank, "Time for scatter I: {:0.4f} in epoch: {}", (time.time() - ticg), epoch)
            prof.append('Scatter I: {:0.4f}'.format(time.time() - ticg))

            tic = time.time()
            ### gather-scatter round II
            for j in range(lim):
                tsend = 0;  trecv = 0
                stn_fp2 = fstorage_comm_nodes_async_fp22.pop()
                out_size_nodes = out_size_nodes_ar[j]

                for i in range(num_parts):
                    tsend += out_size_nodes[i]
                    trecv += stn_fp2[i]

                recv_list_nodes_ = recv_list_nodes[j]
                sten_ = torch.empty(tsend * (feat_size + 1), dtype=feat.dtype)
                dten_ = torch.empty(trecv * (feat_size + 1), dtype=feat.dtype)

                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                offset = 0
                for i in range(num_parts):
                    fdrpa_gather_emb_v42(feat, feat.shape[0], sten_, offset,
                                         recv_list_nodes_[i], out_size_nodes[i],
                                         in_degs, feat_size, i,
                                         node_map_t, num_parts)

                    out_size[i]       = stn_fp2[i] * (feat_size + 1)
                    in_size[i]        = out_size_nodes[i] * (feat_size + 1)
                    offset           += in_size[i]

                req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)
                gfqueue_fp2.push(req)
                ## push dten
                fp2_fstorage_comm_feats_async.push(dten_)
                fp2_fstorage_comm_feats_chunk_async.push(out_size)

            fp2_fstorage_comm_iter.push(lim)

            #message(rank, "Time for gather 2: {:0.4f}",(time.time() - tic))
            prof.append('Gather II: {:0.4f}'.format(time.time() - tic))

            toc = time.time()
            # if epoch >= 2*nrounds_update or nrounds == 0:
            #if epoch - fpass_open >= 2*nrounds_update:
            if epoch - fpass_open >= 2*delay:
                ticg = time.time()

                lim = fp2_fstorage_comm_iter.pop()
                for i in range(lim):
                    tic = time.time()

                    req = gfqueue_fp2.pop()
                    req.wait()

                    #message(rank, "Time for async comms II: {:4f}", (time.time() - tic))
                    prof.append('Async comms II: {:0.4f}'.format(time.time() - tic))

                    otf = fp2_fstorage_comm_feats_async.pop()
                    out_size = fp2_fstorage_comm_feats_chunk_async.pop()
                    stn = fstorage_comm_nodes_async_fp2.pop()

                    offset = 0
                    for l in range(num_parts):
                        assert out_size[l] / (feat_size + 1) == stn[l].shape[0]
                        fdrpa_scatter_reduce_v42(otf, offset, stn[l], stn[l].shape[0],
                                           in_degs, feat, node_map_t, out_size[l], feat_size,
                                           num_parts, rank)

                        offset += out_size[l]

                #message(rank, "Time for scatter 2: {:0.4f}, roundn: {}", time.time() - ticg, roundn)
                prof.append('Scatter II: {:0.4f}'.format(time.time() - ticg))

                #ctx.save_for_backward(node2part, node2part_index, inner_node, node_map_t, adj, in_degs)

                #if rank == 0:
                #    print("Selected nodes: ", selected_nodes_t[roundn].shape, flush=True)

        if rank == 0:
            print(prof, flush=True)
            #print()

        ## for mfg backpass
        #selected_nodes_t = []
        #for i in range(nrounds_):
        #    sn = selected_nodes_bck[epochi]
        #    selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))        
            
        ctx.save_for_backward(node_map_t, adj, lftensor)
        ## added delay variable here
        ctx.backward_cache = epoch, delay, nrounds_update, nrounds,\
            feat_size, base_chunk_size_fs, num_parts, rank, dist,\
            selected_nodes_t[roundn], epochi, nlayers

        #return feat_copy
        return feat
        
    ## R->L; exact working codes: revised BBs and orig BBs backedup below. 
    @staticmethod
    def backward(ctx, grad_out):
        if backpass == 0:
            return grad_out, None, None, None, None, None, None, None, None, None,\
                None, None, None, None, None, None, None, None
        #grad_out_copy = grad_out.clone()
        
        epoch, delay, nrounds_update, nrounds, feat_size, base_chunk_size_fs,\
            num_parts, rank, dist, selected_nodes, epochi, nlayers = ctx.backward_cache
        node_map_t, adj, lftensor = ctx.saved_tensors
        
        if rank == 0:
            print(" >>>>>>>>>>>> drpa core FN BLR bck revised...{}".format(epochi))
        ticg = time.time()
        
        prof = []
        int_threshold = pow(2, 31)/4 - 1              ##bytes
        roundn =  epoch % nrounds if nrounds > 0 else 0
        nrounds_update = nrounds
        width = adj.shape[1]
        
        if epochi == nlayers - 1:
            tic = time.time()
            ## section I: prepare the msg
            buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)         
            num_sel_nodes = torch.zeros(1, dtype=torch.int32)
            sel_nodes = torch.empty(selected_nodes.shape[0] * (num_parts - 1), dtype=torch.int32)
            node2part = torch.empty(selected_nodes.shape[0] * (num_parts - 1), dtype=torch.int32)
            node2part_index = torch.empty(selected_nodes.shape[0] * (num_parts - 1), dtype=torch.int32)
            bdrpa_get_buckets_v6(adj, selected_nodes, sel_nodes, node2part, node2part_index,
                                 node_map_t, buckets, lftensor, num_sel_nodes, width, num_parts,
                                 rank)
            toc = time.time()
            prof.append('[{}] preprocess: {:0.4f}'.format(roundn, toc - tic))

            ###### comms to gather the bucket sizes for all-to-all feats
            input_sr = []
            for i in range(0, num_parts):
                input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))
            
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
            sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated
            
            ### 3. gather emdeddings
            send_feat_len = 0
            in_size = []
            for i in range(num_parts):
                in_size.append(int(buckets[i]) * (feat_size))
                send_feat_len += in_size[i]
            
            ## mpi call split starts
            ##############################################################################
            #tic = time.time()            
            cum = 0; flg = 0
            for i in output_sr:
                cum += int(i) * (feat_size)
                if int(i) >= base_chunk_size_fs: flg = 1
            
            for i in input_sr:
                flg = 1 if int(i) >= base_chunk_size_fs else None
            
            back_off = 1
            if cum >= int_threshold or send_feat_len >= int_threshold or flg:
                for i in range(num_parts):
                    val = ceil((int(input_sr[i]) ) / base_chunk_size_fs)
                    back_off = val if val > back_off else back_off

            tback_off = torch.tensor(back_off)
            rreq = torch.distributed.all_reduce(tback_off, op=torch.distributed.ReduceOp.MAX, \
                                                async_op=True)
            #toc = time.time()
            #prof.append('preprocess: {:0.4f}'.format(toc - tic))
            tic = time.time()
            
            lim = 1
            soffset_base = [0 for i in range(num_parts)]
            soffset_cur = [0 for i in range(num_parts)]       ## send list of ints
            roffset_cur = [0 for i in range(num_parts)]       ## recv list of intrs

            j=0
            while j < lim:
                tsend = 0
                trecv = 0
                for i in range(num_parts):
                    soffset_base[i] += soffset_cur[i]
                    if input_sr[i] < base_chunk_size_fs:
                        soffset_cur[i] = int(input_sr[i])
                        input_sr[i] = 0
                    else:
                        soffset_cur[i] = base_chunk_size_fs
                        input_sr[i] -= base_chunk_size_fs

                    if output_sr[i]  < base_chunk_size_fs:
                        roffset_cur[i] = int(output_sr[i])
                        output_sr[i] = 0
                    else:
                        roffset_cur[i] = base_chunk_size_fs
                        output_sr[i] -= base_chunk_size_fs

                    tsend += soffset_cur[i]
                    trecv += roffset_cur[i]

                send_node_list  = [torch.empty(soffset_cur[i], dtype=torch.int32) for i in range(num_parts)]
                sten_ = torch.empty(tsend * (feat_size), dtype=grad_out.dtype)
                dten_ = torch.empty(trecv * (feat_size), dtype=grad_out.dtype)
                sten_nodes_ = torch.empty(tsend , dtype=torch.int32)
                dten_nodes_ = torch.empty(trecv , dtype=torch.int32)

                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                out_size_nodes = [0 for i in range(num_parts)]
                in_size_nodes = [0 for i in range(num_parts)]

                offset = 0
                for i in range(num_parts):
                    bdrpa_gather_emb_v61(grad_out, grad_out.shape[0], adj, sten_, offset,
                                         send_node_list[i], sten_nodes_,
                                         sel_nodes, node2part, node2part_index,
                                         int(num_sel_nodes[0]), width, feat_size, i,
                                         soffset_base[i], soffset_cur[i], node_map_t, num_parts)

                    out_size[i]       = roffset_cur[i] * (feat_size)
                    in_size[i]        = soffset_cur[i] * (feat_size)
                    offset            += soffset_cur[i]
                    out_size_nodes[i] = roffset_cur[i]
                    in_size_nodes[i]  = soffset_cur[i]

                req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)
                gbqueue.push(req)
                req2 = dist.all_to_all_single(dten_nodes_, sten_nodes_,
                                             out_size_nodes, in_size_nodes,
                                             async_op=True)
                gbqueue2.push(req2)

                #soffset_cur_copy = soffset_cur.copy()
    	        ## section III: store pointers for the data in motion
                bstorage_comm_feats_async.push(dten_)
                bstorage_comm_feats_async2.push(dten_nodes_)
                bstorage_comm_feats_chunk_async.push(out_size)
                bstorage_comm_feats_chunk_async2.push(out_size_nodes)
                #bstorage_comm_nodes_async_fp2.push(send_node_list)    ## fwd phase II
                #bstorage_comm_nodes_async_fp22.push(soffset_cur_copy) ## fwd phase II

                if j == 0:
                    rreq.wait()
                    lim = int(tback_off)
                j += 1
            bstorage_comm_iter.push(lim)
            toc = time.time()
            prof.append('Gather (R->L): {:0.4f}'.format(toc - tic))            
            ##############################################################################
            
            ## section IV: recv the msg and update the aggregates
            recv_list_nodes = []
            tic = time.time()
            if gbqueue.empty() == True:
                print("backward pass Error: epoch: ", epoch, " Empty queue !!!")

            lim = bstorage_comm_iter.pop()
            #bstorage_comm_iter2.push(lim)
            out_size_nodes_ar = []
            for i in range(lim):
                tic_ = time.time()                
                req = gbqueue.pop()
                req.wait()
                req = gbqueue2.pop()
                req.wait()
                toc_ = time.time()
                prof.append('Asycn I: {:0.4f}'.format(toc_ - tic_))

                otf = bstorage_comm_feats_async.pop()
                out_size = bstorage_comm_feats_chunk_async.pop()
                otn = bstorage_comm_feats_async2.pop()
                out_size_nodes = bstorage_comm_feats_chunk_async2.pop()
                #out_size_nodes_ar.append(out_size_nodes)
                #bstorage_out_size_nodes.push(out_size_nodes)

                ilen = 0
                recv_list_nodes_ar = []
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))

                pos = torch.tensor([0], dtype=torch.int64)
                offsetf = 0
                offsetn = 0
                for l in range(num_parts):
                    bdrpa_scatter_reduce_v61(otf, offsetf, otn, offsetn,
                                             grad_out,
                                             node_map_t, out_size[l], feat_size,
                                             num_parts, recv_list_nodes_ar[l], pos,
                                             int(out_size_nodes[l]), rank)

                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]

                assert ilen == pos[0], "Issue in scatter reduce!"
                #recv_list_nodes.append(recv_list_nodes_ar)
                #bstorage_recv_list_nodes.push(recv_list_nodes_ar)
                
            toc = time.time()
            prof.append('Scatter I: {:0.4f}'.format(toc - tic))
            
        else:
            
            tic = time.time()
            ### gather-scatter round II
            assert bstorage_comm_iter2.size() == 1
            lim = bstorage_comm_iter2.pop()
            # assert lim == 1
            for j in range(lim):
                tsend = 0
                trecv = 0
                stn_fp2 = bstorage_comm_nodes_async_fp22.pop()
                # out_size_nodes = out_size_nodes_ar[j]
                out_size_nodes = bstorage_out_size_nodes.pop()
            
                for i in range(num_parts):
                    tsend += out_size_nodes[i]
                    trecv += stn_fp2[i]
            
                # recv_list_nodes_ = recv_list_nodes[j]
                recv_list_nodes_ = bstorage_recv_list_nodes.pop()
            
                sten_ = torch.empty(tsend * (feat_size), dtype=grad_out.dtype)
                dten_ = torch.empty(trecv * (feat_size), dtype=grad_out.dtype)
            
                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                offset = 0
                for i in range(num_parts):
                    ## gather by leafs
                    bdrpa_gather_emb_v42(grad_out, grad_out.shape[0], sten_, offset,
                                         recv_list_nodes_[i], out_size_nodes[i],
                                         feat_size, i,
                                         node_map_t, num_parts)
            
                    out_size[i]       = stn_fp2[i] * (feat_size)
                    in_size[i]        = out_size_nodes[i] * (feat_size)
                    offset           += in_size[i]
            
                req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)
                gbqueue_fp2.push(req)
            
                ## push dten
                fp2_bstorage_comm_feats_async.push(dten_)
                fp2_bstorage_comm_feats_chunk_async.push(out_size)
            
            fp2_bstorage_comm_iter.push(lim)
            
            toc = time.time()
            prof.append('Gather II: {:0.4f}'.format(toc - tic))            
            
            ##### recv and scatter
            tic = time.time()            
            lim = fp2_bstorage_comm_iter.pop()
            for i in range(lim):
                tic_ = time.time()
                req = gbqueue_fp2.pop()
                req.wait()
                toc_ = time.time()
                prof.append('Async II: {:0.4f}'.format(toc_ - tic_))

                otf = fp2_bstorage_comm_feats_async.pop()
                out_size = fp2_bstorage_comm_feats_chunk_async.pop()
                stn = bstorage_comm_nodes_async_fp2.pop()

                offset = 0
                for l in range(num_parts): ## R->L
                    assert out_size[l] / (feat_size) == stn[l].shape[0]
                    #bdrpa_scatter_reduce_v62(otf, offset, stn[l], stn[l].shape[0],
                    #                         grad_out, node_map_t, out_size[l], feat_size,
                    #                         num_parts, rank)
                    bdrpa_scatter_reduce_v42(otf, offset, stn[l], stn[l].shape[0],
                                             grad_out, node_map_t, out_size[l], feat_size,
                                             num_parts, rank)

                    offset += out_size[l]
            toc = time.time()
            prof.append('Scatter II: {:0.4f}'.format(toc - tic))

        ## zero out the leaf node gradients
        #rootify2(grad_out, grad_out.shape[0], grad_out.shape[1], root_node)
        if rank == 0:
            tocg = time.time()
            print(prof, flush=True)
            print("Remote agg L->R time: {:.4f}".format(tocg - ticg))
        
        return grad_out, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None, None, None
        #return grad_out_copy, None, None, None, None, None, None, None, None, None,\
        #    None, None, None, None, None, None, None



        
## 2nd half call -- after local aggregation
def call_drpa_core_FF_BRL(neigh, adj,inner_node, lftensor, selected_nodes, selected_nodes_bck,
                          node_map, num_parts,
                          rank, epoch, dist, degs, delay, nrounds, output_sr_ar, gather_q41, epochi,
                          nlayers, flag):
    return drpa_core_FF_BRL.apply(neigh, adj, inner_node, lftensor,
                                  selected_nodes, selected_nodes_bck,
                                  node_map, num_parts, rank,
                                  epoch, dist, degs, delay, nrounds, output_sr_ar,
                                  gather_q41, epochi, nlayers, flag)



## 1st drpa call
class drpa_core_FN_BLR(torch.autograd.Function):
    # FN, BLR
    @staticmethod
    def forward(ctx, feat_, adj_, inner_node_, lftensor_, selected_nodes,
                node_map, num_parts, rank, epoch, dist, in_degs, delay, nrounds,
                output_sr_ar, gather_q41, epochi, nlayers, root_node):

        #feat_copy = feat.clone()        
        if rank == 0:
            print(" >>>>>>>>>>>> drpa core FN BLR...{}".format(epochi))
            print("drpa_core_last: ", epochi, flush=True)

        nrounds_update = nrounds

        feat = feat_
        adj = adj_
        inner_node = inner_node_
        lftensor = lftensor_
        feat_size = feat.shape[1]
        int_threshold = pow(2, 31)/4 - 1              ##bytes
        base_chunk_size = int(int_threshold / num_parts) ##bytes
        base_chunk_size_fs = floor(base_chunk_size / (feat_size + 1) )
        node_map_t = torch.tensor(node_map, dtype=torch.int32)
        selected_nodes_t = []

        nrounds_ = 1 if nrounds == 0 else nrounds
        #for sn in selected_nodes[epochi]:
        for i in range(nrounds_):
            #### sn = selected_nodes[epochi]
            sn = selected_nodes[epochi - 1]   ##mfg
            selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))
            # print("fwd 1 sel size: ", epochi, " ", selected_nodes_t[i].size())
            
        roundn =  epoch % nrounds if nrounds > 0 else 0

        ctx.save_for_backward(node_map_t, adj, lftensor, root_node)
        ## added delay variable here
        ctx.backward_cache = epoch, delay, nrounds_update, nrounds, \
            feat_size, base_chunk_size_fs, num_parts, rank, dist,\
            selected_nodes_t[roundn], epochi, nlayers

        return feat

    ## L -> R
    @staticmethod
    def backward(ctx, grad_out):
        if backpass == 0:
            return grad_out, None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None, None, None, None
        #grad_out_copy = grad_out.clone()
        epoch, delay, nrounds_update, nrounds, feat_size, base_chunk_size_fs, \
            num_parts, rank, dist, selected_nodes, epochi, nlayers = ctx.backward_cache
        node_map_t, adj, lftensor, root_node = ctx.saved_tensors
        
        if rank == 0:
            print(" >>>>>>>>>>>> drpa core FF BRL bck revised...{}".format(epochi), flush=True)        

        ticg = time.time()
        prof = []
        int_threshold = pow(2, 31)/4 - 1              ##bytes
        roundn =  epoch % nrounds  if nrounds > 0 else 0
        nrounds_update = nrounds
        width = adj.shape[1]

        if epoch >= bpass_open and (epoch >= 0*nrounds_update or nrounds == 0):
            ## print("selected nodes: ", selected_nodes)
            ##"""
            tic = time.time()
            """
            ## section I: prepare the msg
            buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)         
            node2part = torch.empty(selected_nodes.shape[0], dtype=torch.int32)
            node2part_index = torch.empty(selected_nodes.shape[0], dtype=torch.int32)
            #bdrpa_get_buckets_v6(adj, selected_nodes, sel_nodes, node2part, node2part_index,
            #                     node_map_t, buckets, lftensor, num_sel_nodes, width, num_parts,
            #                     rank)
            fdrpa_get_buckets_v4(adj, selected_nodes, node2part, node2part_index,
                                 node_map_t, buckets, lftensor, width, num_parts, rank)
            
            #toc = time.time()
            #prof.append('[{}] preprocess: {:0.4f}'.format(roundn, toc - tic))
            ###### comms to gather the bucket sizes for all-to-all feats
            input_sr = []
            for i in range(0, num_parts):
                input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))
            
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
            sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated
            """

            obj = bpre_aux[epochi]
            #back_off = drpa_comm_iters(obj.output_sr_t, obj.input_sr_t, feat_size,
            #                           base_chunk_size_fs, num_parts, int(int_threshold))
            node2part = obj.node2part
            node2part_index = obj.node2part_index
            input_sr = obj.input_sr.copy()
            output_sr = obj.output_sr.copy()
            buckets = obj.buckets
            ## end of new code which had replaced the above commented code

            ### 3. gather emdeddings
            """
            send_feat_len = 0
            in_size = []
            for i in range(num_parts):
                in_size.append(int(buckets[i]) * (feat_size))
                send_feat_len += in_size[i]
            """
            in_size = (buckets * feat_size).tolist()
            send_feat_len = int(buckets.sum()) * feat_size
            #assert send_feat_len == send_feat_len_
            #assert in_size == in_size_
            
            toc = time.time()
            prof.append('[{}] preprocess: {:0.4f}'.format(roundn, toc - tic))
                
            ## mpi call split starts
            ##############################################################################
            tic = time.time()            
            cum = 0; flg = 0
            for i in output_sr:
                cum += int(i) * (feat_size)
                if int(i) >= base_chunk_size_fs: flg = 1
            
            for i in input_sr:
                flg = 1 if int(i) >= base_chunk_size_fs else None
            
            back_off = 1
            if cum >= int_threshold or send_feat_len >= int_threshold or flg:
                for i in range(num_parts):
                    val = ceil((int(input_sr[i]) ) / base_chunk_size_fs)
                    back_off = val if val > back_off else back_off

            tback_off = torch.tensor(back_off)
            rreq = torch.distributed.all_reduce(tback_off, op=torch.distributed.ReduceOp.MAX, \
                                                async_op=True)
            toc = time.time()
            prof.append('[{}] preprocess2: {:0.4f}'.format(roundn, toc - tic))

            tic = time.time()
            
            lim = 1
            soffset_base = [0 for i in range(num_parts)]
            soffset_cur = [0 for i in range(num_parts)]       ## send list of ints
            roffset_cur = [0 for i in range(num_parts)]       ## recv list of intrs

            j=0
            while j < lim:
                tsend = 0
                trecv = 0
                for i in range(num_parts):
                    soffset_base[i] += soffset_cur[i]
                    if input_sr[i] < base_chunk_size_fs:
                        soffset_cur[i] = int(input_sr[i])
                        input_sr[i] = 0
                    else:
                        soffset_cur[i] = base_chunk_size_fs
                        input_sr[i] -= base_chunk_size_fs

                    if output_sr[i]  < base_chunk_size_fs:
                        roffset_cur[i] = int(output_sr[i])
                        output_sr[i] = 0
                    else:
                        roffset_cur[i] = base_chunk_size_fs
                        output_sr[i] -= base_chunk_size_fs

                    tsend += soffset_cur[i]
                    trecv += roffset_cur[i]

                send_node_list  = [torch.empty(soffset_cur[i], dtype=torch.int32) for i in range(num_parts)]
                sten_ = torch.empty(tsend * (feat_size), dtype=grad_out.dtype)
                dten_ = torch.empty(trecv * (feat_size), dtype=grad_out.dtype)
                sten_nodes_ = torch.empty(tsend , dtype=torch.int32)
                dten_nodes_ = torch.empty(trecv , dtype=torch.int32)

                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                out_size_nodes = [0 for i in range(num_parts)]
                in_size_nodes = [0 for i in range(num_parts)]

                offset = 0
                for i in range(num_parts):
                    #bdrpa_gather_emb_v61(grad_out, grad_out.shape[0], adj, sten_, offset,
                    #                     send_node_list[i], sten_nodes_,
                    #                     sel_nodes, node2part, node2part_index,
                    #                     int(num_sel_nodes[0]), width, feat_size, i,
                    #                     soffset_base[i], soffset_cur[i], node_map_t, num_parts)
                    bdrpa_gather_emb_v41(grad_out, grad_out.shape[0], adj, sten_, offset,
                                         send_node_list[i], sten_nodes_,
                                         selected_nodes,
                                         node2part, node2part_index, width, feat_size, i,
                                         soffset_base[i], soffset_cur[i], node_map_t, num_parts)

                    out_size[i]       = roffset_cur[i] * (feat_size)
                    in_size[i]        = soffset_cur[i] * (feat_size)
                    offset            += soffset_cur[i]
                    out_size_nodes[i] = roffset_cur[i]
                    in_size_nodes[i]  = soffset_cur[i]

                req = dist.all_to_all_single(dten_, sten_, out_size, in_size, async_op=True)
                gbqueue.push(req)
                req2 = dist.all_to_all_single(dten_nodes_, sten_nodes_,
                                             out_size_nodes, in_size_nodes,
                                             async_op=True)
                gbqueue2.push(req2)

                soffset_cur_copy = soffset_cur.copy()
    	        ## section III: store pointers for the data in motion
                bstorage_comm_feats_async.push(dten_)
                bstorage_comm_feats_async2.push(dten_nodes_)
                bstorage_comm_feats_chunk_async.push(out_size)
                bstorage_comm_feats_chunk_async2.push(out_size_nodes)
                bstorage_comm_nodes_async_fp2.push(send_node_list)    ## fwd phase II
                bstorage_comm_nodes_async_fp22.push(soffset_cur_copy) ## fwd phase II

                if j == 0:
                    rreq.wait()
                    lim = int(tback_off)
                j += 1
            bstorage_comm_iter.push(lim)
            toc = time.time()
            prof.append('Gather (R->L): {:0.4f}'.format(toc - tic))            
        ##############################################################################
        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []
        if epoch >= bpass_open and (epoch >= 0*nrounds_update or nrounds == 0):
            tic = time.time()
            if gbqueue.empty() == True:
                print("backward pass Error: epoch: ", epoch, " Empty queue !!!")

            lim = bstorage_comm_iter.pop()
            bstorage_comm_iter2.push(lim)
            out_size_nodes_ar = []
            for i in range(lim):
                tic_ = time.time()                
                req = gbqueue.pop()
                req.wait()
                req = gbqueue2.pop()
                req.wait()
                toc_ = time.time()
                prof.append('Asycn I: {:0.4f}'.format(toc_ - tic_))

                otf = bstorage_comm_feats_async.pop()
                out_size = bstorage_comm_feats_chunk_async.pop()
                otn = bstorage_comm_feats_async2.pop()
                out_size_nodes = bstorage_comm_feats_chunk_async2.pop()
                #out_size_nodes_ar.append(out_size_nodes)
                bstorage_out_size_nodes.push(out_size_nodes)

                ilen = 0
                recv_list_nodes_ar = []
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))

                pos = torch.tensor([0], dtype=torch.int64)
                offsetf = 0
                offsetn = 0
                for l in range(num_parts): ## L->R
                    #bdrpa_scatter_reduce_v61(otf, offsetf, otn, offsetn,
                    #                         grad_out,
                    #                         node_map_t, out_size[l], feat_size,
                    #                         num_parts, recv_list_nodes_ar[l], pos,
                    #                         int(out_size_nodes[l]), rank)
                    bdrpa_scatter_reduce_v41(otf, offsetf, otn, offsetn,
                                             grad_out, node_map_t, out_size[l], feat_size,
                                             num_parts, recv_list_nodes_ar[l], pos,
                                             int(out_size_nodes[l]), rank)

                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]

                assert ilen == pos[0], "Issue in scatter reduce!"
                #recv_list_nodes.append(recv_list_nodes_ar)
                bstorage_recv_list_nodes.push(recv_list_nodes_ar)
                
            toc = time.time()
            prof.append('Scatter I: {:0.4f}'.format(toc - tic))

        tic = time.time()
        rootify2(grad_out, grad_out.shape[0], grad_out.shape[1], root_node)
        prof.append('Rootify: {:0.4f}'.format(time.time() - tic))
        if rank == 0:
            tocg = time.time()
            print(prof, flush=True)
            print("Remote agg R->L time: {:.4f}".format(tocg - ticg))
            
        return grad_out, None, None, None, None, None, None, None, None, \
            None, None, None, None, None, None, None, None, None
        #return grad_out_copy, None, None, None, None, None, None, None, None, \
        #    None, None, None, None, None, None, None, None

                        

## drpa first half call
def call_drpa_core_FN_BLR(neigh, adj,inner_node, lftensor, selected_nodes, node_map, num_parts,
                          rank, epoch, dist, degs, delay, nrounds, output_sr_ar, gather_q41, epochi,
                          nlayers, flag):
    return drpa_core_FN_BLR.apply(neigh, adj, inner_node, lftensor,
                                  selected_nodes, node_map, num_parts, rank,
                                  epoch, dist, degs, delay, nrounds, output_sr_ar,
                                  gather_q41, epochi, nlayers, flag)

