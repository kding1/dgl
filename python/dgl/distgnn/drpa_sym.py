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
from ..sparse import bdrpa_scatter_reduce_v41, bdrpa_gather_emb_v41, bdrpa_scatter_reduce_v42, bdrpa_gather_emb_v42, bdrpa_grad_normalize

#z1=122565
#z2=46134
#z3=17365
#z4=107340
#z5=11666
#z6=111983
#z7=44034

z1=236191
z2=44708
z3=17459
z4=42216
z5=210700
z6=57478
z7=103756


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
fp2_bstorage_comm_feats_async = gqueue()
fp2_bstorage_comm_feats_chunk_async = gqueue()
fp2_bstorage_comm_iter = gqueue()
bstorage_comm_nodes_async_fp22_ext = gqueue()


## DRPA
def drpa(gobj, rank, num_parts, node_map, nrounds, dist, nlayers):
    #d = drpa_master(gobj, rank, num_parts, node_map, nrounds, dist, nlayers)
    d = drpa_master(gobj._graph, gobj._ntypes, gobj._etypes, gobj._node_frames, gobj._edge_frames)
    d.drpa_init(rank, num_parts, node_map, nrounds, dist, nlayers)
    return d

class drpa_master(DGLHeteroGraph):
    def drpa_init(self, rank, num_parts, node_map, nrounds, dist, nlayers):
        #print("In drpa Init....")
        self.rank = rank
        self.num_parts = num_parts
        self.node_map = node_map
        self.nrounds = nrounds
        self.dist = dist
        self.nlayers = nlayers + 1

        self.epochs_ar = [0 for i in range(self.nlayers)]
        self.epochi = 0
        self.gather_q41 = gqueue()
        self.output_sr_ar = []

        if self.nrounds == -1: return
        ## Creates buckets based on ndrounds

        adj = self.dstdata['adj']
        lf = self.dstdata['lf']
        width = adj.shape[1]
        #print("width: ", width)
        ## groups send data according to 'send to rank'
        ## communicates and fill output_sr_ar for data to be recd.
        self.drpa_create_buckets()
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

    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        assert self.rank != -1, "drpa not initialized !!!"
        #print("feats: ", self.dstdata['h'])
        epoch = self.epochs_ar[self.epochi]
        if True and self.nrounds != -1:
            #if self.rank == 0:
            #    print("update_all self.epochi: ", self.epochi, flush=True)
            feat_dst = self.dstdata['h']
            self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
            neigh = self.dstdata['h']
            adj = self.dstdata['adj']
            inner_node = self.dstdata['inner_node']
            lftensor = self.dstdata['lf']
            feat_dst = self.dstdata['h']

           #self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
            self.dstdata['h'] = call_drpa_core_last(neigh, adj, inner_node,
                                                    lftensor, self.selected_nodes,
                                                    self.node_map, self.num_parts,
                                                    self.rank, epoch,
                                                    self.dist,
                                                    self.r_in_degs,
                                                    self.nrounds,
                                                    self.output_sr_ar,
                                                    self.gather_q41,
                                                    self.epochi,
                                                    self.nlayers,
                                                    0)
            self.srcdata['h'] = self.dstdata['h']

        #print("rfunc: ", reduce_func.name)
        mean = 0
        if reduce_func.name == "mean":
            reduce_func = fn.sum('m', 'neigh')
            mean = 1

        #print("rfunc: ", reduce_func.name)

        tic = time.time()
        DGLHeteroGraph.update_all(self, message_func, reduce_func)
        toc = time.time()

        #if self.rank == 0:
        #    torch.save(self.dstdata['neigh'], "r1-l1-feature.pt")
        #    torch.save(self.ndata['inner_node'], "r1-l1-inner_node.pt")
        #    torch.save(self.dstdata['orig'], "r1-l1-orig.pt")
        #    print("node0 in_edges: ", self.in_edges(0))
        #    node0 = self.ndata['feat'][1] + self.ndata['feat'][2] + self.ndata['feat'][3]
        #    print("agg for 0: ", node0[19], " ", self.dstdata['neigh'][0][19])
        #else:
        #    torch.save(self.dstdata['neigh'], "r2-l1-feature.pt")
        #    torch.save(self.ndata['inner_node'], "r2-l1-inner_node.pt")
        #    torch.save(self.dstdata['orig'], "r2-l1-orig.pt")
        #print("Exiting....")
        #exit()

        if self.rank == 0  and display:
            print("Time for local aggregate: {:0.4f}, nrounds {}".format(toc - tic, self.nrounds))

        if self.nrounds == -1:
            if mean == 1:
                feat_dst = self.dstdata['h']
                self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
                self.dstdata['neigh'] = self.dstdata['neigh'] / (self.r_in_degs.unsqueeze(-1) + 1)
            return

        neigh = self.dstdata['neigh']
        adj = self.dstdata['adj']
        inner_node = self.dstdata['inner_node']
        lftensor = self.dstdata['lf']
        feat_dst = self.dstdata['h']
        epoch = self.epochs_ar[self.epochi]

        tic = time.time()
        
        self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
        self.dstdata['neigh'] = call_drpa_core(neigh, adj, inner_node, lftensor, self.selected_nodes,
                                               self.node_map, self.num_parts, self.rank, epoch,
                                               self.dist,
                                               self.r_in_degs,
                                               self.nrounds,
                                               self.output_sr_ar,
                                               self.gather_q41,
                                               self.epochi,
                                               self.nlayers,
                                               1)

        #self.epochs_ar[self.epochi] += 1
        #self.epochi = (self.epochi + 1) % (self.nlayers)
            
        toc = time.time()
        if self.rank == 0 and display:
            print("Time for remote aggregate: {:0.4f}".format(toc - tic))

        if mean == 1:
            #self.dstdata['neigh'] = self.dstdata['neigh'] / (self.r_in_degs.unsqueeze(-1) + 1)
            self.dstdata['neigh'] = self.dstdata['neigh'] / (self.r_in_degs.unsqueeze(-1))

        #if self.rank == 0:  ##debug
        #    print("drpa degs: ")
        #    orig = self.ndata['orig']
        #    for i in range(10):
        #        #print("{} orig: {}, deg: {}".format(i, orig[i], self.r_in_degs[i]))
        #        print("agg: ", self.dstdata['neigh'][i])
        #        print("adj: ", adj[i], " shape: ", adj[i].shape[0])
            

    def loss_grad_sync(self, neigh):
        #print("loss grad sync....", flush=True)
        epoch = self.epochs_ar[self.epochi]
        self.r_in_degs = DGLHeteroGraph.in_degrees(self).to(neigh)
        adj = self.dstdata['adj']
        inner_node = self.dstdata['inner_node']
        lftensor = self.dstdata['lf']

        grads = call_drpa_core_loss_sync(neigh, adj, inner_node,
                                         lftensor, self.selected_nodes,
                                         self.node_map, self.num_parts,
                                         self.rank, epoch,
                                         self.dist,
                                         self.r_in_degs,
                                         self.nrounds,
                                         self.output_sr_ar,
                                         self.gather_q41,
                                         self.epochi,
                                         self.nlayers,
                                         0)

        self.epochs_ar[self.epochi] += 1
        self.epochi = (self.epochi + 1) % (self.nlayers)
        return grads
        
        
    def drpa_init_buckets(self, adj, lf, width):
        nm = torch.tensor(self.node_map, dtype=torch.int32)
        for l in range(self.nrounds):
            buckets = torch.tensor([0 for i in range(self.num_parts)], dtype=torch.int32)
            ## fill buckets here
            #print("nrounds: ", self.nrounds)
            #print("l: ", l)
            #print("selected_nodes: ", len(self.selected_nodes[l]))
            sn = torch.tensor(self.selected_nodes[l], dtype=torch.int32)
            fdrpa_init_buckets_v4(adj, sn, nm, buckets,
                                  lf, width, self.num_parts, self.rank)
            input_sr = []
            for i in range(0, self.num_parts):
                input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))

            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, self.num_parts)]
            sync_req = self.dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated
            self.output_sr_ar.append(output_sr)  ## output


    def drpa_create_buckets(self):
        inner_nodex = self.ndata['inner_node'].tolist() ##.count(1)
        n = len(inner_nodex)
        idx = inner_nodex.count(1)

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


    def in_degrees(self):
        try:
            return self.r_in_degs
        except:
            print("Passing only local node degrees.")
            pass

        return DGLHeteroGraph.in_degrees(self)


    #def drpa_accumulate_part_degs(self):
    #    adj = self.dstdata['adj']
    #    inner_node = self.dstdata['inner_node']
    #    lftensor = self.dstdata['lf']
    #    feat_dst = self.dstdata['h']
    #    in_degs = DGLHeteroGraph.in_degrees(self).to(feat_dst)
    #    nm = torch.tensor(self.node_map, dtype=torch.int32)
    #    width = adj.shape[1]
    #
    #    self.full_n_degs = in_degs
    #
    #    #nsend_nodes_t = torch.tensor([0 for i in range(self.num_parts)], dtype= torch.int32)
    #    nsend_nodes_t = torch.zeros(self.num_parts, dtype= torch.int32)
    #    drpa_num_send_nodes(adj, inner_node, nm, nsend_nodes_t, width, self.rank, self.num_parts)
    #
    #    nsend_nodes = []
    #    nrecv_nodes = []
    #    for i in range(0, self.num_parts):
    #        nsend_nodes.append(torch.tensor([nsend_nodes_t[i]], dtype=torch.int32))
    #        nrecv_nodes.append(torch.zeros(1, dtype=torch.int32))
    #
    #    req = self.dist.all_to_all(nrecv_nodes, nsend_nodes, async_op=True)
    #
    #    sindex = []
    #    sdeg = []
    #    for i in range(0, self.num_parts):
    #        sindex.append(torch.zeros(nsend_nodes[i], dtype=torch.int64))
    #        sdeg.append(torch.zeros(nsend_nodes[i], dtype=torch.int64))
    #        if i == p:
    #            assert int(nsend_nodes[i]) == 0, "nsend to myseld nonzero"
    #        else:
    #            drpa_populate_deg(adj, inner_node, nm, in_degs, sindex[i], sdeg[i], width, i, self.num_parts)
    #
    #    req.wait()
    #
    #    rindex = []
    #    rdeg = []
    #    for i in range(0, self.num_parts):
    #        rindex.append(torch.zeros(nrecv_nodes[i], dtype=torch.int64))
    #        rdeg.append(torch.zeros(nrecv_nodes[i], dtype=torch.int64))
    #
    #
    #    req1 = self.dist.all_to_all(rindex, sindex, async_op=True)
    #    req2 = self.dist.all_to_all(rdeg, sdeg, async_op=True)
    #    req1.wait()
    #    req2.wait()
    #
    #    for i in range(0, self.num_parts):
    #        ri = rindex[i]
    #        rd = rdeg[i]
    #        assert ri.shape[0] == rd.shape[0], "ri and rd shape not matching!!"
    #        for j in range(ri.shape[0]):
    #            self.full_in_deg[ri[j]] += rd[j]

def message(rank, msg, val=-1, val_=-1):
    if rank == 0 and display:
        if val == -1:
            print(msg, flush=True)
        elif val_ == -1:
            print(msg.format(val), flush=True)
        else:
            print(msg.format(val, val_), flush=True)


class drpa_core(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat, adj, inner_node, lftensor, selected_nodes,
                node_map, num_parts, rank, epoch, dist, in_degs, nrounds,
                output_sr_ar, gather_q41, epochi, nlayers, pass_flag):

        #if pass_flag == 0:
        #    flag = 1
        #    ctx.backward_cache2 = flag
        #    return feat     ## dummy fwdpass
        
        nrounds_update = nrounds
        #if rank == 0 and display:
        #    print(" >>>>>>>>>>>> drpa code fwd...{}".format(epochi))

        prof = []
        feat_size = feat.shape[1]
        int_threshold = pow(2, 31)/4 - 1              ##bytes
        base_chunk_size = int(int_threshold / num_parts) ##bytes
        base_chunk_size_fs = floor(base_chunk_size / (feat_size + 1) )
        roundn =  epoch % nrounds

        node_map_t = torch.tensor(node_map, dtype=torch.int32)
        selected_nodes_t = []
        for sn in selected_nodes:
            selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))

        ## section I: prepare the msg
        buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        width = adj.shape[1]

        tic = time.time()

        #### 1. get bucket sizes
        node2part = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        node2part_index = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        fdrpa_get_buckets_v4(adj, selected_nodes_t[roundn], node2part, node2part_index,
                             node_map_t, buckets, lftensor, width, num_parts, rank)

        #message(rank, "Time for get buckets: {:0.4f}", (time.time() - tic))

        ###### comms to gather the bucket sizes for all-to-all feats
        input_sr = []
        for i in range(0, num_parts):
            input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))

        if False:
            output_sr = output_sr_ar[roundn].copy()   ## computed during drpa_init
        else:
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
            sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated

        input_sr_copy = input_sr.copy()
        output_sr_copy = output_sr.copy()

        ### 3. gather emdeddings
        send_feat_len = 0
        in_size = []
        for i in range(num_parts):
            in_size.append(int(buckets[i]) * (feat_size + 1))
            send_feat_len += in_size[i]

        ## mpi call split starts
        ##############################################################################
        tic = time.time()

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

        tback_off = torch.tensor(back_off)
        rreq = torch.distributed.all_reduce(tback_off, op=torch.distributed.ReduceOp.MAX, async_op=True)

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
        prof.append('Gather I: {:0.4f}'.format(time.time() - tic))

        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []
        if epoch >= nrounds_update or nrounds == 1:
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
            prof.append('Scatter I: {:0.4f}'.format(time.time() - tic))

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
            if epoch >= 2*nrounds_update or nrounds == 1:
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

        ctx.save_for_backward(node2part, node2part_index, inner_node, node_map_t, adj, in_degs)
        ctx.backward_cache = input_sr_copy, output_sr_copy, epoch, nrounds_update, nrounds,\
            feat_size, base_chunk_size_fs, lim, num_parts, rank, dist,\
            selected_nodes_t[roundn], epochi, nlayers
        flag = 0  ## flip the flag
        ctx.backward_cache2 = flag

        #if rank == 0:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat0: ", feat[55])
        #    print("node map: ", node_map_t)
        #    print("adj 10: ", adj[55], " lf: ", lftensor[55])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat1: ", feat[z1])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat2: ", feat[z2])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat3: ", feat[z3])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat4: ", feat[z4])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat5: ", feat[z5])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat6: ", feat[z6])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat7: ", feat[z7])

        #if rank == 0:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat0: ", feat[10])
        #    print("node map: ", node_map_t)
        #    print("adj 10: ", adj[10])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat1: ", feat[11843])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat2: ", feat[13847])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat3: ", feat[28099])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat4: ", feat[17385])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat5: ", feat[16465])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat6: ", feat[1479])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig fwdpass, feat7: ", feat[13665])
        
        return feat


    ## Master drpa backward
    @staticmethod
    def backward(ctx, grad_out):
        #input_sr, output_sr, epoch, nrounds_update, nrounds, feat_size, base_chunk_size_fs,\
        #    lim, num_parts, rank, dist, selected_nodes, epochi, nlayers = ctx.backward_cache
        #node2part, node2part_index, inner_node, node_map_t, adj, in_degs = ctx.saved_tensors
        #if rank == 0:
        #    #flag = 10
        #    #for i in range(grad_out.shape[0]):
        #    #    if adj[i].tolist().count(-1) == 7 and flag >= 0:
        #    #        flag -= 1
        #    #        print("Epoch: ", epoch ,"orig backpass, grad_out0 >>: ", grad_out[i])
        #    print("Epoch: ", epoch ,"orig backpass, grad_out0 >>: ", grad_out[0])
        #    print("adj 0: ", adj[0])
        #    print("Epoch: ", epoch ,"orig backpass, grad_out0: ", grad_out[55])
        #    print("node map: ", node_map_t)
        #    print("adj 10: ", adj[55])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig backpass, grad_out1: ", grad_out[z1])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig backpass, grad_out2: ", grad_out[z2])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig backpass, grad_out3: ", grad_out[z3])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig backpass, grad_out4: ", grad_out[z4])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig backpass, grad_out5: ", grad_out[z5])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig backpass, grad_out6: ", grad_out[z6])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig backpass, grad_out7: ", grad_out[z7])
        
        #if rank == 0:
        #    print("Epoch: ", epoch ,"orig backpass, grad0: ", grad_out[10])
        #    print("node map: ", node_map_t)
        #    print("adj 10: ", adj[10])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig backpass, grad1: ", grad_out[11843])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig backpass, grad2: ", grad_out[13847])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig backpass, grad3: ", grad_out[28099])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig backpass, grad4: ", grad_out[17385])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig backpass, grad5: ", grad_out[16465])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig backpass, grad6: ", grad_out[1479])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig backpass, grad7: ", grad_out[13665])

        return grad_out, None, None, None, None, None, None, None, None, None, \
            None, None, None, None, None, None, None
    
        #flag = ctx.backward_cache2
        #if flag == 0:
        #    return grad_out, None, None, None, None, None, None, None, None, None, \
        #        None, None, None, None, None, None, None
                    
        input_sr, output_sr, epoch, nrounds_update, nrounds, feat_size, base_chunk_size_fs,\
            lim, num_parts, rank, dist, selected_nodes, epochi, nlayers = ctx.backward_cache

        if epochi != 2:
            return grad_out, None, None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None
        
        if rank == 0:
            print(" >>>>>>>>>>>> drpa code bck...{}".format(epochi))

        node2part, node2part_index, inner_node, node_map_t, adj, in_degs = ctx.saved_tensors

        #if epochi == nlayers - 1:
        #    if rank == 0:
        #        print("Level-1")
        #    grad_out = grad_out / (in_degs.unsqueeze(-1) + 1)
        #    return grad_out, None, None, None, None, None, None, None,\
        #        None, None, None, None, None, None, None, None


        if epoch < 2*nrounds_update and nrounds != 1:
            return grad_out, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None, None, None

        #if rank == 0 and display:
        #    print()
        #    #print("rank: {}, nrounds: {}, epoch: {}, lim: {}, feat_size: {}".format(rank, nrounds, epoch, lim, feat_size))
        #    #print("B input_sr: ", input_sr)
        #    #print("B output_sr: ", output_sr)
       #    #print("B Selected nodes: ", selected_nodes.shape)
        #    print(flush=True)
        #
        ##node2part, node2part_index, inner_node, node_map_t, adj, in_degs = ctx.saved_tensors

        roundn =  epoch % nrounds
        nrounds_update = nrounds
        width = adj.shape[1]

        if epoch >= 2*nrounds_update or nrounds == 1:
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
                    ## gather by followers
                    #dist.barrier()
                    #if rank == 0:
                    #    print(">i: ", i, flush=True)
                    bdrpa_gather_emb_v41(grad_out, grad_out.shape[0], adj, sten_, offset,
                                         send_node_list[i], sten_nodes_,
                                         selected_nodes,
                                         node2part, node2part_index, width, feat_size, i,
                                         soffset_base[i], soffset_cur[i], node_map_t, num_parts)
                    #dist.barrier()
                    #if rank == 0:
                    #    print("<i: ", i, flush=True)

                    out_size[i]       = roffset_cur[i] * (feat_size)
                    in_size[i]        = soffset_cur[i] * (feat_size)
                    offset            += soffset_cur[i]
                    out_size_nodes[i] = roffset_cur[i]
                    in_size_nodes[i]  = soffset_cur[i]


                if rank == 0 and display:
                    print("B Sending {}, recving {} data I".format(tsend, trecv), flush=True)

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

                #if j == 0:
                #    rreq.wait()
                #    lim = int(tback_off)
                j += 1
        ##############################################################################
        ## mpi call split ends
        if rank == 0  and display:
            print("B Max iters in MPI split comm: {}".format(lim), flush=True)

        bstorage_comm_iter.push(lim)

        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []
        if epoch >= 3*nrounds_update or nrounds == 1:
            if gbqueue.empty() == True:
                print("backward pass Error: epoch: ", epoch, " Empty queue !!!")
            #if rank == 0 and display:
            ticg = time.time()

            lim = bstorage_comm_iter.pop()
            out_size_nodes_ar = []
            for i in range(lim):
                if rank == 0 and display:
                    tic = time.time()

                req = gbqueue.pop()
                req.wait()
                req = gbqueue2.pop()
                req.wait()

                if rank == 0 and display:
                    print("B Time for async comms I: {:4f}".format(time.time() - tic), flush=True)

                otf = bstorage_comm_feats_async.pop()
                out_size = bstorage_comm_feats_chunk_async.pop()
                otn = bstorage_comm_feats_async2.pop()
                out_size_nodes = bstorage_comm_feats_chunk_async2.pop()
                out_size_nodes_ar.append(out_size_nodes)

                ilen = 0
                recv_list_nodes_ar = []
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))

                pos = torch.tensor([0], dtype=torch.int64)
                offsetf = 0
                offsetn = 0
                for l in range(num_parts):
                    bdrpa_scatter_reduce_v41(otf, offsetf, otn, offsetn,
                                             grad_out,
                                             node_map_t, out_size[l], feat_size,
                                             num_parts, recv_list_nodes_ar[l], pos,
                                             int(out_size_nodes[l]), rank)

                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]

                assert ilen == pos[0], "Issue in scatter reduce!"
                recv_list_nodes.append(recv_list_nodes_ar)

            ##bdrpa_grad_normalize(grad_out, adj, selected_nodes, feat_size, num_parts, rank, width)

            tocg = time.time()
            if rank == 0 and display:
                print("B Time for scatter I: {:0.4f} in epoch: {}".format(tocg - ticg, epoch), flush=True)

            #if rank == 0 and display:
            tic = time.time()

            ### gather-scatter round II
            for j in range(lim):
                tsend = 0
                trecv = 0
                stn_fp2 = bstorage_comm_nodes_async_fp22.pop()
                out_size_nodes = out_size_nodes_ar[j]

                for i in range(num_parts):
                    tsend += out_size_nodes[i]
                    trecv += stn_fp2[i]

                recv_list_nodes_ = recv_list_nodes[j]

                sten_ = torch.empty(tsend * (feat_size), dtype=grad_out.dtype)
                dten_ = torch.empty(trecv * (feat_size), dtype=grad_out.dtype)


                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                offset = 0
                for i in range(num_parts):
                    ## gather by leader
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
            if rank == 0 and display:
                print("B Time for gather 2: {:0.4f}".format(toc - tic), flush=True)

        if epoch >= 4*nrounds_update or nrounds == 1:
            #if rank == 0 and display:
            ticg = time.time()
            if rank == 0 and display:
                print("B epoch: {} - scatter_reduce_v41, lim: {}".format(epoch, lim))

            lim = fp2_bstorage_comm_iter.pop()
            for i in range(lim):
                if rank == 0 and display:
                    tic = time.time()

                req = gbqueue_fp2.pop()
                req.wait()

                if rank == 0 and display:
                    print("B Time for async comms II: {:4f}".format(time.time() - tic), flush=True)

                otf = fp2_bstorage_comm_feats_async.pop()
                out_size = fp2_bstorage_comm_feats_chunk_async.pop()
                stn = bstorage_comm_nodes_async_fp2.pop()

                offset = 0
                for l in range(num_parts):
                    assert out_size[l] / (feat_size) == stn[l].shape[0]
                    bdrpa_scatter_reduce_v42(otf, offset, stn[l], stn[l].shape[0],
                                             grad_out, node_map_t, out_size[l], feat_size,
                                             num_parts, rank)

                    offset += out_size[l]

            tocg = time.time()
            if rank == 0 and display:
                print("B Time for scatter 2: {:0.4f}".format(tocg - ticg))

        ## if rank == 0:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out0: ", grad_out[55])
        ##     print("node map: ", node_map_t)
        ##     print("adj 10: ", adj[55])
        ## if rank == 1:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out1: ", grad_out[z1])
        ## if rank == 2:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out2: ", grad_out[z2])
        ## if rank == 3:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out3: ", grad_out[z3])
        ## if rank == 4:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out4: ", grad_out[z4])
        ## if rank == 5:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out5: ", grad_out[z5])
        ## if rank == 6:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out6: ", grad_out[z6])
        ## if rank == 7:
        ##     print("Epoch: ", epoch ,"orig backpass12, grad_out7: ", grad_out[z7])        
                
        return grad_out, None, None, None, None, None, None, None, None, None,\
            None, None, None, None, None, None, None



def call_drpa_core(neigh, adj,inner_node, lftensor, selected_nodes, node_map, num_parts,
                   rank, epoch, dist, degs, nrounds, output_sr_ar, gather_q41, epochi,
                   nlayers, flag):
    return drpa_core.apply(neigh, adj, inner_node, lftensor,
                           selected_nodes, node_map, num_parts, rank,
                           epoch, dist, degs, nrounds, output_sr_ar,
                           gather_q41, epochi, nlayers, flag)




class degree_division__(torch.autograd.Function):
    #def __init__():

    @staticmethod
    def forward(ctx, neigh, h, feat_size, degs, lim):
        #print("in forward pass of deg_div", flush=True)
        deg_div(neigh, h, feat_size, degs, lim)
        ctx.backward_cache = feat_size, lim
        ctx.save_for_backward(degs)
        return neigh

    @staticmethod
    def backward(ctx, grad_out):
        #print("in backward pass of deg_div", flush=True)

        feat_size, lim = ctx.backward_cache
        degs = ctx.saved_tensors[0]
        deg_div_back(grad_out, feat_size, degs, lim)

        return grad_out, None, None, None, None


def deg_div_class(neigh, h, feat_size, degs, lim):
    return degree_division.apply(neigh, h, feat_size, degs, lim)


class drpa_core_last(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_, adj_, inner_node_, lftensor_, selected_nodes,
                node_map, num_parts, rank, epoch, dist, in_degs, nrounds,
                output_sr_ar, gather_q41, epochi, nlayers, flag):

        if rank == 0:
            print(" >>>>>>>>>>>> drpa code last fwd...{}".format(epochi))

        if rank == 0:
            print("drpa_core_last: ", epochi, flush=True)
        #if epochi != 1:
        #    ctx.backward_cache2 = epochi
        #    return feat_

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
        for sn in selected_nodes:
            selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))

        roundn =  epoch % nrounds

        ## section I: prepare the msg
        buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        width = adj.shape[1]

        num_sel_nodes = torch.tensor([0], dtype=torch.int32)
        cnt = 0

        if rank == 0 and display:
            print("FWD pass: nrounds: ", nrounds, flush=True)

        if rank == 0:
            tic = time.time()

        #### 1. get bucket sizes
        node2part = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        node2part_index = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        fdrpa_get_buckets_v4(adj, selected_nodes_t[roundn], node2part, node2part_index,
                             node_map_t, buckets, lftensor,
                             width, num_parts, rank)


        if rank == 0  and display:
            toc = time.time()
            print("Time for get buckets: {:0.4f} in epoch: {}".format(toc - tic, epoch))


        ###### comms to gather the bucket sizes for all-to-all feats
        input_sr = []
        #output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
        for i in range(0, num_parts):
            input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))


        if False:
            output_sr = output_sr_ar[roundn].copy()   ## computed during drpa_init
        else:
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
            sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated

        input_sr_copy = input_sr.copy()
        output_sr_copy = output_sr.copy()
        ### 3. gather emdeddings
        send_feat_len = 0
        in_size = []
        for i in range(num_parts):
            in_size.append(int(buckets[i]) * (feat_size + 1))
            send_feat_len += in_size[i]

        ## mpi call split starts
        ##############################################################################
        #if rank == 0 and display:
        tic = time.time()

        cum = 0
        flg = 0
        for i in output_sr:
            cum += int(i) * (feat_size + 1)
            if int(i) >= base_chunk_size_fs: flg = 1

        for i in input_sr:
            if int(i) >= base_chunk_size_fs: flg = 1

        back_off = 1
        if cum >= int_threshold or send_feat_len >= int_threshold or flg:
            for i in range(num_parts):
                val = ceil((int(input_sr[i]) ) / base_chunk_size_fs)
                if val > back_off:
                    back_off = val

        #print("back_off: {}".format(back_off), flush=True)
        tback_off = torch.tensor(back_off)
        rreq = torch.distributed.all_reduce(tback_off, op=torch.distributed.ReduceOp.MAX, async_op=True)

        #lim = int(tback_off)  ## async req waited at the bottom of the loop
        lim = 1

        rreq.wait()
        lim = int(tback_off)

        ctx.save_for_backward(node2part, node2part_index, inner_node, node_map_t, adj, in_degs)
        ctx.backward_cache = input_sr_copy, output_sr_copy, epoch, nrounds_update, nrounds, \
            feat_size, base_chunk_size_fs, lim, num_parts, rank, dist, \
            selected_nodes_t[roundn], nlayers
        ctx.backward_cache2 = epochi, rank

        return feat



    @staticmethod
    def backward(ctx, grad_out):
        
        #return grad_out, None, None, None, None, None, None, None, None, \
        #    None, None, None, None, None, None, None
        epochi, rank_ = ctx.backward_cache2

        if rank_ == 0:
            print(" >>>>>>>>>>>> drpa code last bck...{}".format(epochi))

        input_sr, output_sr, epoch, nrounds_update, nrounds, feat_size, base_chunk_size_fs, \
            lim, num_parts, rank, dist, selected_nodes, nlayers = ctx.backward_cache        
        node2part, node2part_index, inner_node, node_map_t, adj, in_degs = ctx.saved_tensors

        ## if rank == 0:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out0: ", grad_out[55])
        ##     print("node map: ", node_map_t)
        ##     print("adj 10: ", adj[55])
        ## if rank == 1:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out1: ", grad_out[z1])
        ## if rank == 2:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out2: ", grad_out[z2])
        ## if rank == 3:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out3: ", grad_out[z3])
        ## if rank == 4:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out4: ", grad_out[z4])
        ## if rank == 5:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out5: ", grad_out[z5])
        ## if rank == 6:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out6: ", grad_out[z6])
        ## if rank == 7:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out7: ", grad_out[z7])        
        #if rank == 0:
        #    print("Epoch: ", epoch ,"orig backpass2, grad0: ", grad_out[10])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig backpass2, grad1: ", grad_out[11843])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig backpass2, grad2: ", grad_out[13847])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig backpass2, grad3: ", grad_out[28099])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig backpass2, grad4: ", grad_out[17385])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig backpass2, grad5: ", grad_out[16465])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig backpass2, grad6: ", grad_out[1479])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig backpass2, grad7: ", grad_out[13665])
        
        #if rank == 0:
        #    print("lim: {}, num_parts {}, nround: {}, epoch {}, feat_size {}".
        #          format(lim, num_parts, nrounds, epoch, feat_size))

        if epoch < 2*nrounds_update and nrounds != 1:
            return grad_out, None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None

        if rank == 0 and display:
            print()
            print(flush=True)

        #node2part, node2part_index, inner_node, node_map_t, adj, in_degs = ctx.saved_tensors

        roundn =  epoch % nrounds
        nrounds_update = nrounds
        width = adj.shape[1]

        if epoch >= 2*nrounds_update or nrounds == 1:
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
                    ## gather by followers
                    #dist.barrier()
                    #if rank == 0:
                    #    print(">i: ", i, flush=True)
                    bdrpa_gather_emb_v41(grad_out, grad_out.shape[0], adj, sten_, offset,
                                         send_node_list[i], sten_nodes_,
                                         selected_nodes,
                                         node2part, node2part_index, width, feat_size, i,
                                         soffset_base[i], soffset_cur[i], node_map_t, num_parts)
                    #dist.barrier()
                    #if rank == 0:
                    #    print("<i: ", i, flush=True)

                    out_size[i]       = roffset_cur[i] * (feat_size)
                    in_size[i]        = soffset_cur[i] * (feat_size)
                    offset            += soffset_cur[i]
                    out_size_nodes[i] = roffset_cur[i]
                    in_size_nodes[i]  = soffset_cur[i]


                #if rank == 0 and display:
                #    print("B Sending {}, recving {} data I".format(tsend, trecv), flush=True)

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

                #if j == 0:
                #    rreq.wait()
                #    lim = int(tback_off)
                j += 1
        ##############################################################################
        ## mpi call split ends
        #if rank == 0  and display:
        #    print("B Max iters in MPI split comm: {}".format(lim), flush=True)

        bstorage_comm_iter.push(lim)

        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []
        if epoch >= 3*nrounds_update or nrounds == 1:
            if gbqueue.empty() == True:
                print("backward pass Error: epoch: ", epoch, " Empty queue !!!")
            #if rank == 0 and display:
            ticg = time.time()

            lim = bstorage_comm_iter.pop()
            out_size_nodes_ar = []
            for i in range(lim):
                if rank == 0 and display:
                    tic = time.time()

                req = gbqueue.pop()
                req.wait()
                req = gbqueue2.pop()
                req.wait()

                #if rank == 0 and display:
                #    print("B Time for async comms I: {:4f}".format(time.time() - tic), flush=True)

                otf = bstorage_comm_feats_async.pop()
                out_size = bstorage_comm_feats_chunk_async.pop()
                otn = bstorage_comm_feats_async2.pop()
                out_size_nodes = bstorage_comm_feats_chunk_async2.pop()
                out_size_nodes_ar.append(out_size_nodes)

                ilen = 0
                recv_list_nodes_ar = []
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))

                pos = torch.tensor([0], dtype=torch.int64)
                offsetf = 0
                offsetn = 0
                for l in range(num_parts):
                    bdrpa_scatter_reduce_v41(otf, offsetf, otn, offsetn,
                                             grad_out,
                                             node_map_t, out_size[l], feat_size,
                                             num_parts, recv_list_nodes_ar[l], pos,
                                             int(out_size_nodes[l]), rank)

                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]

                assert ilen == pos[0], "Issue in scatter reduce!"
                recv_list_nodes.append(recv_list_nodes_ar)

            ##bdrpa_grad_normalize(grad_out, adj, selected_nodes, feat_size, num_parts, rank, width)

            tocg = time.time()
            #if rank == 0 and display:
            #    print("B Time for scatter I: {:0.4f} in epoch: {}".format(tocg - ticg, epoch), flush=True)

            #if rank == 0 and display:
            tic = time.time()

            ### gather-scatter round II
            for j in range(lim):
                tsend = 0
                trecv = 0
                stn_fp2 = bstorage_comm_nodes_async_fp22.pop()
                out_size_nodes = out_size_nodes_ar[j]

                for i in range(num_parts):
                    tsend += out_size_nodes[i]
                    trecv += stn_fp2[i]

                recv_list_nodes_ = recv_list_nodes[j]

                sten_ = torch.empty(tsend * (feat_size), dtype=grad_out.dtype)
                dten_ = torch.empty(trecv * (feat_size), dtype=grad_out.dtype)


                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                offset = 0
                for i in range(num_parts):
                    ## gather by leader
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
            #if rank == 0 and display:
            #    print("B Time for gather 2: {:0.4f}".format(toc - tic), flush=True)

        if epoch >= 4*nrounds_update or nrounds == 1:
            #if rank == 0 and display:
            ticg = time.time()
            #if rank == 0 and display:
            #    print("B epoch: {} - scatter_reduce_v41, lim: {}".format(epoch, lim))

            lim = fp2_bstorage_comm_iter.pop()
            for i in range(lim):
                if rank == 0 and display:
                    tic = time.time()

                req = gbqueue_fp2.pop()
                req.wait()

                #if rank == 0 and display:
                #    print("B Time for async comms II: {:4f}".format(time.time() - tic), flush=True)

                otf = fp2_bstorage_comm_feats_async.pop()
                out_size = fp2_bstorage_comm_feats_chunk_async.pop()
                stn = bstorage_comm_nodes_async_fp2.pop()

                offset = 0
                for l in range(num_parts):
                    assert out_size[l] / (feat_size) == stn[l].shape[0]
                    bdrpa_scatter_reduce_v42(otf, offset, stn[l], stn[l].shape[0],
                                             grad_out, node_map_t, out_size[l], feat_size,
                                             num_parts, rank)

                    offset += out_size[l]

            tocg = time.time()
            #if rank == 0 and display:
            #    print("B Time for scatter 2: {:0.4f}".format(tocg - ticg))

        ## if rank == 0:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out0: ", grad_out[55])
        ##     print("node map: ", node_map_t)
        ##     print("adj 10: ", adj[55])
        ## if rank == 1:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out1: ", grad_out[z1])
        ## if rank == 2:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out2: ", grad_out[z2])
        ## if rank == 3:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out3: ", grad_out[z3])
        ## if rank == 4:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out4: ", grad_out[z4])
        ## if rank == 5:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out5: ", grad_out[z5])
        ## if rank == 6:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out6: ", grad_out[z6])
        ## if rank == 7:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out7: ", grad_out[z7])                    
        #if rank == 0:
        #    print("Epoch: ", epoch ,"orig backpass3, grad0: ", grad_out[10])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig backpass3, grad1: ", grad_out[11843])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig backpass3, grad2: ", grad_out[13847])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[28099])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[17385])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[16465])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[1479])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[13665])
            
        #print("last backpass exit , grad: ", grad_out[10])
        #print("adj of node 0: ", adj[10])
        #grad_out /= (in_degs.unsqueeze(-1) + 1)
        return grad_out, None, None, None, None, None, None, None, None, \
            None, None, None, None, None, None, None, None



def call_drpa_core_last(neigh, adj,inner_node, lftensor, selected_nodes, node_map, num_parts,
                        rank, epoch, dist, degs, nrounds, output_sr_ar, gather_q41, epochi, nlayers,
                        flag):
    return drpa_core_last.apply(neigh, adj, inner_node, lftensor,
                                selected_nodes, node_map, num_parts, rank,
                                epoch, dist, degs, nrounds, output_sr_ar,
                                gather_q41, epochi, nlayers, flag)




class drpa_core_loss_sync(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_, adj_, inner_node_, lftensor_, selected_nodes,
                node_map, num_parts, rank, epoch, dist, in_degs, nrounds,
                output_sr_ar, gather_q41, epochi, nlayers, flag):

        if rank == 0:
            print(" >>>>>>>>>>>> loss sync drpa core last fwd...{}".format(epochi))
            #print("rst: ", feat_)

        if rank == 0:
            print("drpa_core_last: ", epochi, flush=True)
        #if epochi != 1:
        #    ctx.backward_cache2 = epochi
        #    return feat_

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
        for sn in selected_nodes:
            selected_nodes_t.append(torch.tensor(sn, dtype=torch.int32))

        roundn =  epoch % nrounds

        ## section I: prepare the msg
        buckets = torch.tensor([0 for i in range(num_parts)], dtype=torch.int32)
        width = adj.shape[1]

        num_sel_nodes = torch.tensor([0], dtype=torch.int32)
        cnt = 0

        if rank == 0 and display:
            print("FWD pass: nrounds: ", nrounds, flush=True)

        if rank == 0:
            tic = time.time()

        #### 1. get bucket sizes
        node2part = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        node2part_index = torch.empty(len(selected_nodes[roundn]), dtype=torch.int32)
        fdrpa_get_buckets_v4(adj, selected_nodes_t[roundn], node2part, node2part_index,
                             node_map_t, buckets, lftensor,
                             width, num_parts, rank)


        if rank == 0  and display:
            toc = time.time()
            print("Time for get buckets: {:0.4f} in epoch: {}".format(toc - tic, epoch))


        ###### comms to gather the bucket sizes for all-to-all feats
        input_sr = []
        #output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
        for i in range(0, num_parts):
            input_sr.append(torch.tensor([buckets[i]], dtype=torch.int64))


        if False:
            output_sr = output_sr_ar[roundn].copy()   ## computed during drpa_init
        else:
            output_sr = [torch.zeros(1, dtype=torch.int64) for i in range(0, num_parts)]
            sync_req = dist.all_to_all(output_sr, input_sr, async_op=True)   # make it async
            sync_req.wait() ## recv the #nodes communicated

        input_sr_copy = input_sr.copy()
        output_sr_copy = output_sr.copy()
        ### 3. gather emdeddings
        send_feat_len = 0
        in_size = []
        for i in range(num_parts):
            in_size.append(int(buckets[i]) * (feat_size + 1))
            send_feat_len += in_size[i]

        ## mpi call split starts
        ##############################################################################
        #if rank == 0 and display:
        tic = time.time()

        cum = 0
        flg = 0
        for i in output_sr:
            cum += int(i) * (feat_size + 1)
            if int(i) >= base_chunk_size_fs: flg = 1

        for i in input_sr:
            if int(i) >= base_chunk_size_fs: flg = 1

        back_off = 1
        if cum >= int_threshold or send_feat_len >= int_threshold or flg:
            for i in range(num_parts):
                val = ceil((int(input_sr[i]) ) / base_chunk_size_fs)
                if val > back_off:
                    back_off = val

        #print("back_off: {}".format(back_off), flush=True)
        tback_off = torch.tensor(back_off)
        rreq = torch.distributed.all_reduce(tback_off, op=torch.distributed.ReduceOp.MAX, async_op=True)

        #lim = int(tback_off)  ## async req waited at the bottom of the loop
        lim = 1

        rreq.wait()
        lim = int(tback_off)

        ctx.save_for_backward(node2part, node2part_index, inner_node, node_map_t, adj, in_degs)
        ctx.backward_cache = input_sr_copy, output_sr_copy, epoch, nrounds_update, nrounds, \
            feat_size, base_chunk_size_fs, lim, num_parts, rank, dist, \
            selected_nodes_t[roundn], nlayers
        ctx.backward_cache2 = epochi, rank

        if rank == 0  and display:
            print("return feat....")
        return feat
    
        
    @staticmethod
    def backward(ctx, grad_out):
        #return grad_out, None, None, None, None, None, None, None, None, \
        #    None, None, None, None, None, None, None
        epochi, rank_ = ctx.backward_cache2 
       
        if epochi != 2:
            return grad_out, None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None, None, None

        if rank_ == 0:
            print(" >>>>>>>>>>>> loss sync drpa code last bck...{}".format(epochi))

        input_sr, output_sr, epoch, nrounds_update, nrounds, feat_size, base_chunk_size_fs, \
            lim, num_parts, rank, dist, selected_nodes, nlayers = ctx.backward_cache        
        node2part, node2part_index, inner_node, node_map_t, adj, in_degs = ctx.saved_tensors

        ## if rank == 0:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out0: ", grad_out[55])
        ##     print("node map: ", node_map_t)
        ##     print("adj 10: ", adj[55])
        ## if rank == 1:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out1: ", grad_out[z1])
        ## if rank == 2:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out2: ", grad_out[z2])
        ## if rank == 3:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out3: ", grad_out[z3])
        ## if rank == 4:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out4: ", grad_out[z4])
        ## if rank == 5:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out5: ", grad_out[z5])
        ## if rank == 6:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out6: ", grad_out[z6])
        ## if rank == 7:
        ##     print("Epoch: ", epoch ,"orig backpass2, grad_out7: ", grad_out[z7])        
        #if rank == 0:
        #    print("Epoch: ", epoch ,"orig backpass2, grad0: ", grad_out[10])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig backpass2, grad1: ", grad_out[11843])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig backpass2, grad2: ", grad_out[13847])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig backpass2, grad3: ", grad_out[28099])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig backpass2, grad4: ", grad_out[17385])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig backpass2, grad5: ", grad_out[16465])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig backpass2, grad6: ", grad_out[1479])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig backpass2, grad7: ", grad_out[13665])
        
        #if rank == 0:
        #    print("lim: {}, num_parts {}, nround: {}, epoch {}, feat_size {}".
        #          format(lim, num_parts, nrounds, epoch, feat_size))

        if epoch < 2*nrounds_update and nrounds != 1:
            return grad_out, None, None, None, None, None, None, None, None, \
                None, None, None, None, None, None

        if rank == 0 and display:
            print()
            print(flush=True)

        #node2part, node2part_index, inner_node, node_map_t, adj, in_degs = ctx.saved_tensors

        roundn =  epoch % nrounds
        nrounds_update = nrounds
        width = adj.shape[1]

        if epoch >= 2*nrounds_update or nrounds == 1:
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
                    ## gather by followers
                    #dist.barrier()
                    #if rank == 0:
                    #    print(">i: ", i, flush=True)
                    bdrpa_gather_emb_v41(grad_out, grad_out.shape[0], adj, sten_, offset,
                                         send_node_list[i], sten_nodes_,
                                         selected_nodes,
                                         node2part, node2part_index, width, feat_size, i,
                                         soffset_base[i], soffset_cur[i], node_map_t, num_parts)
                    #dist.barrier()
                    #if rank == 0:
                    #    print("<i: ", i, flush=True)

                    out_size[i]       = roffset_cur[i] * (feat_size)
                    in_size[i]        = soffset_cur[i] * (feat_size)
                    offset            += soffset_cur[i]
                    out_size_nodes[i] = roffset_cur[i]
                    in_size_nodes[i]  = soffset_cur[i]


                #if rank == 0 and display:
                #    print("B Sending {}, recving {} data I".format(tsend, trecv), flush=True)

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

                #if j == 0:
                #    rreq.wait()
                #    lim = int(tback_off)
                j += 1
        ##############################################################################
        ## mpi call split ends
        #if rank == 0  and display:
        #    print("B Max iters in MPI split comm: {}".format(lim), flush=True)

        bstorage_comm_iter.push(lim)

        ## section IV: recv the msg and update the aggregates
        recv_list_nodes = []
        if epoch >= 3*nrounds_update or nrounds == 1:
            if gbqueue.empty() == True:
                print("backward pass Error: epoch: ", epoch, " Empty queue !!!")
            #if rank == 0 and display:
            ticg = time.time()

            lim = bstorage_comm_iter.pop()
            out_size_nodes_ar = []
            for i in range(lim):
                if rank == 0 and display:
                    tic = time.time()

                req = gbqueue.pop()
                req.wait()
                req = gbqueue2.pop()
                req.wait()

                #if rank == 0 and display:
                #    print("B Time for async comms I: {:4f}".format(time.time() - tic), flush=True)

                otf = bstorage_comm_feats_async.pop()
                out_size = bstorage_comm_feats_chunk_async.pop()
                otn = bstorage_comm_feats_async2.pop()
                out_size_nodes = bstorage_comm_feats_chunk_async2.pop()
                out_size_nodes_ar.append(out_size_nodes)

                ilen = 0
                recv_list_nodes_ar = []
                for l in range(num_parts):
                    ilen += out_size_nodes[l]
                    recv_list_nodes_ar.append(torch.empty(out_size_nodes[l], dtype=torch.int32))

                pos = torch.tensor([0], dtype=torch.int64)
                offsetf = 0
                offsetn = 0
                for l in range(num_parts):
                    bdrpa_scatter_reduce_v41(otf, offsetf, otn, offsetn,
                                             grad_out,
                                             node_map_t, out_size[l], feat_size,
                                             num_parts, recv_list_nodes_ar[l], pos,
                                             int(out_size_nodes[l]), rank)

                    offsetf += out_size[l]
                    offsetn += out_size_nodes[l]

                assert ilen == pos[0], "Issue in scatter reduce!"
                recv_list_nodes.append(recv_list_nodes_ar)

            ##bdrpa_grad_normalize(grad_out, adj, selected_nodes, feat_size, num_parts, rank, width)

            tocg = time.time()
            #if rank == 0 and display:
            #    print("B Time for scatter I: {:0.4f} in epoch: {}".format(tocg - ticg, epoch), flush=True)

            #if rank == 0 and display:
            tic = time.time()

            ### gather-scatter round II
            for j in range(lim):
                tsend = 0
                trecv = 0
                stn_fp2 = bstorage_comm_nodes_async_fp22.pop()
                out_size_nodes = out_size_nodes_ar[j]

                for i in range(num_parts):
                    tsend += out_size_nodes[i]
                    trecv += stn_fp2[i]

                recv_list_nodes_ = recv_list_nodes[j]

                sten_ = torch.empty(tsend * (feat_size), dtype=grad_out.dtype)
                dten_ = torch.empty(trecv * (feat_size), dtype=grad_out.dtype)


                out_size = [0 for i in range(num_parts)]
                in_size = [0 for i in range(num_parts)]
                offset = 0
                for i in range(num_parts):
                    ## gather by leader
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
            #if rank == 0 and display:
            #    print("B Time for gather 2: {:0.4f}".format(toc - tic), flush=True)

        if epoch >= 4*nrounds_update or nrounds == 1:
            #if rank == 0 and display:
            ticg = time.time()
            #if rank == 0 and display:
            #    print("B epoch: {} - scatter_reduce_v41, lim: {}".format(epoch, lim))

            lim = fp2_bstorage_comm_iter.pop()
            for i in range(lim):
                if rank == 0 and display:
                    tic = time.time()

                req = gbqueue_fp2.pop()
                req.wait()

                #if rank == 0 and display:
                #    print("B Time for async comms II: {:4f}".format(time.time() - tic), flush=True)

                otf = fp2_bstorage_comm_feats_async.pop()
                out_size = fp2_bstorage_comm_feats_chunk_async.pop()
                stn = bstorage_comm_nodes_async_fp2.pop()

                offset = 0
                for l in range(num_parts):
                    assert out_size[l] / (feat_size) == stn[l].shape[0]
                    bdrpa_scatter_reduce_v42(otf, offset, stn[l], stn[l].shape[0],
                                             grad_out, node_map_t, out_size[l], feat_size,
                                             num_parts, rank)

                    offset += out_size[l]

            tocg = time.time()
            #if rank == 0 and display:
            #    print("B Time for scatter 2: {:0.4f}".format(tocg - ticg))

        ## if rank == 0:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out0: ", grad_out[55])
        ##     print("node map: ", node_map_t)
        ##     print("adj 10: ", adj[55])
        ## if rank == 1:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out1: ", grad_out[z1])
        ## if rank == 2:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out2: ", grad_out[z2])
        ## if rank == 3:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out3: ", grad_out[z3])
        ## if rank == 4:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out4: ", grad_out[z4])
        ## if rank == 5:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out5: ", grad_out[z5])
        ## if rank == 6:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out6: ", grad_out[z6])
        ## if rank == 7:
        ##     print("Epoch: ", epoch ,"orig backpass3, grad_out7: ", grad_out[z7])                    
        #if rank == 0:
        #    print("Epoch: ", epoch ,"orig backpass3, grad0: ", grad_out[10])
        #if rank == 1:
        #    print("Epoch: ", epoch ,"orig backpass3, grad1: ", grad_out[11843])
        #if rank == 2:
        #    print("Epoch: ", epoch ,"orig backpass3, grad2: ", grad_out[13847])
        #if rank == 3:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[28099])
        #if rank == 4:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[17385])
        #if rank == 5:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[16465])
        #if rank == 6:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[1479])
        #if rank == 7:
        #    print("Epoch: ", epoch ,"orig backpass3, grad3: ", grad_out[13665])
            
        #print("last backpass exit , grad: ", grad_out[10])
        #print("adj of node 0: ", adj[10])
        #grad_out /= (in_degs.unsqueeze(-1) + 1)
        return grad_out, None, None, None, None, None, None, None, None, \
            None, None, None, None, None, None, None, None



def call_drpa_core_loss_sync(neigh, adj,inner_node, lftensor, selected_nodes, node_map, num_parts,
                        rank, epoch, dist, degs, nrounds, output_sr_ar, gather_q41, epochi, nlayers,
                        flag):
    return drpa_core_loss_sync.apply(neigh, adj, inner_node, lftensor,
                                     selected_nodes, node_map, num_parts, rank,
                                     epoch, dist, degs, nrounds, output_sr_ar,
                                     gather_q41, epochi, nlayers, flag)



#### under-construction and dromant code block below
class drpa_class_update_all(torch.autograd.Function):
    @staticmethod
    def forward(ctx, drpa_master_obj,
                message_func,
                reduce_func):
        mean = 0
        if reduce_func.name == "mean":
            reduce_func = fn.sum('m', 'neigh')
            mean = 1

        tic = time.time()
        DGLHeteroGraph.update_all(drpa_master_obj, message_func, reduce_func)
        toc = time.time()

        if drpa_master_obj.rank == 0  and display:
            print("Time for local aggregate: {:0.4f}, nrounds {}".format(toc - tic, drpa_master_obj.nrounds))

        neigh = drpa_master_obj.dstdata['neigh']
        adj = drpa_master_obj.dstdata['adj']
        inner_node = drpa_master_obj.dstdata['inner_node']
        lftensor = drpa_master_obj.dstdata['lf']
        feat_dst = drpa_master_obj.dstdata['h']
        epoch = drpa_master_obj.epochs_ar[drpa_master_obj.epochi]

        tic = time.time()

        drpa_master_obj.r_in_degs = DGLHeteroGraph.in_degrees(drpa_master_obj).to(feat_dst)
        drpa_master_obj.dstdata['neigh'] = call_drpa_core(neigh, adj, inner_node, lftensor,
                                                          drpa_master_obj.selected_nodes,
                                                          drpa_master_obj.node_map,
                                                          drpa_master_obj.num_parts,
                                                          drpa_master_obj.rank, epoch,
                                                          drpa_master_obj.dist,
                                                          drpa_master_obj.r_in_degs,
                                                          drpa_master_obj.nrounds,
                                                          drpa_master_obj.output_sr_ar,
                                                          drpa_master_obj.gather_q41,
                                                          drpa_master_obj.epochi,
                                                          drpa_master_obj.nlayers)

        drpa_master_obj.epochs_ar[drpa_master_obj.epochi] += 1
        drpa_master_obj.epochi = (drpa_master_obj.epochi + 1) % (drpa_master_obj.nlayers)

        toc = time.time()
        if drpa_master_obj.rank == 0 and display:
            print("Time for remote aggregate: {:0.4f}".format(toc - tic))

        if mean == 1:
            drpa_master_obj.dstdata['neigh'] = \
                drpa_master_obj.dstdata['neigh'] / drpa_master_obj.r_in_degs.unsqueeze(-1)


    #@staticmethod
    #def backward(ctx, grad_out):
