import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import uniform
import torch.distributed as dist
from dgl.sparse import gather_floats, scatter_floats, broadcast_floats

## Python or C/C++ code for scatter-gather
py=False
LOG=False
class dist_batch_norm(nn.Module):

    def __init__(self, in_size, root_nid, rank, momentum=0.9, eps = 1e-5):
        super(dist_batch_norm, self).__init__()

        self.root_nid = root_nid
        self.momentum = momentum
        self.insize = in_size
        self.eps = eps
        self.rank = rank
        
        U = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.gamma = nn.Parameter(U.sample(torch.Size([self.insize])).view(self.insize))
        self.beta = nn.Parameter(torch.zeros(self.insize))            

    def printv(self):
        print("bn file name: ", os.path.basename(__file__), flush=True)
        
    def forward(self, input):
        
        X = input
        X_gat = 0
        send_n = 0

        #tic = time.time()
        #X_gat = gather_emb(X, self.root_nid)
        #print(X_gat)
        #print("gather time: {:.4f}".format(time.time() - tic))

        if LOG:
            tic = time.time()
        #mean = bn1d_mean(X, self.root_nid, self.rank)
        X_gat = torch.empty([self.root_nid.shape[0], X.shape[1]], dtype=X.dtype)
        mean = bn1d_mean(X, self.root_nid, X_gat)
        if LOG:
            print("mean time: {:.4f}".format(time.time() - tic))

        if LOG:
            tic = time.time()            
        variance = bn1d_var(X, mean, self.root_nid, X_gat)
        if LOG:
            print("var time: {:.4f} ".format(time.time() - tic))

        if LOG:
            tic = time.time()
        X_hat = (X-mean) * 1.0 /torch.sqrt(variance + self.eps)
        out = self.gamma * X_hat + self.beta
        if LOG:
            print("xhat time: {:.4f} ".format(time.time() - tic))
        
        return out
        # return input


class dist_bn_mean(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, X, root_nid, rank):
    def forward(ctx, X, root_nid, X_gat):
        #tic = time.time()
        """
        if py:
            print("Gather code: Python")
            root_nid_index = root_nid.unsqueeze(1).expand(root_nid.shape[0], X.shape[1])
            X_gat = torch.gather(X, 0, root_nid_index)
        else:
            X_gat = torch.empty([root_nid.shape[0], X.shape[1]], dtype=X.dtype)
            gather_floats(X, root_nid, X_gat)
        """

        gather_floats(X, root_nid, X_gat)
        send_n = torch.tensor(X_gat.shape[0], dtype=torch.int64)
        sum_send = torch.sum(X_gat, axis=0)
        
        ## Allreduce 1
        req = dist.all_reduce(sum_send, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        req = dist.all_reduce(send_n, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        mean = sum_send / send_n
        
        ctx.backward_cache = X.shape[0], send_n
        ctx.save_for_backward(root_nid)
        
        return mean

    @staticmethod
    def backward(ctx, grad_mean):    
        size, send_n = ctx.backward_cache
        root_nid, = ctx.saved_tensors
        feat_size = grad_mean.shape[0]

        if LOG:
            tic = time.time()        
        req = dist.all_reduce(grad_mean, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        grad_mean /= send_n
        #tic = time.time()
        if py:
            grad_out = torch.zeros([size, feat_size], dtype=grad_mean.dtype)
            index = root_nid.unsqueeze(1)
            index = index.expand(root_nid.shape[0], feat_size)
            grad_out.scatter_(0, index, grad_mean.expand(index.shape[0], feat_size))
        else:
            grad_out = torch.zeros([size, feat_size], dtype=grad_mean.dtype)
            broadcast_floats(grad_mean, root_nid, grad_out)

        if LOG:            
            print("mean scatter(bck) time: ", time.time() - tic)
        return grad_out, None, None


def bn1d_mean(X, root_nid, X_gat):
    return dist_bn_mean.apply(X, root_nid, X_gat)


class dist_bn_var(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, X, mean, send_n, X_gat, root_nid):
    def forward(ctx, X, mean, root_nid, X_gat):
        #tic = time.time()
        """
        if py:
            root_nid_index = root_nid.unsqueeze(1).expand(root_nid.shape[0], X.shape[1])
            X_gat = torch.gather(X, 0, root_nid_index)
        else:
            X_gat = torch.empty([root_nid.shape[0], X.shape[1]], dtype=X.dtype)
            gather_floats(X, root_nid, X_gat)        
        #print("var gather time: ", time.time() - tic)
        """
        
        # sum_send = torch.sum(X_gat, axis=0)
        send_n = torch.tensor(X_gat.shape[0], dtype=torch.int64)

        ## Allreduce 1
        req = dist.all_reduce(send_n, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        var_send = torch.sum((X_gat - mean)**2, axis=0)
        req = dist.all_reduce(var_send, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        variance = var_send / send_n

        ctx.backward_cache = X.shape[0], send_n
        ctx.save_for_backward(X_gat, root_nid)
        return variance

    @staticmethod
    def backward(ctx, grad_var):    
        size, send_n = ctx.backward_cache
        X_gat, root_nid = ctx.saved_tensors
        feat_size = grad_var.shape[0]
        
        if LOG:
            tic = time.time()
        
        req = dist.all_reduce(grad_var, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        grad_var = grad_var.unsqueeze(0) * 2 * X_gat / send_n
        #tic = time.time()
        if py:
            grad_out = torch.zeros([size, feat_size], dtype=grad_var.dtype)
            index = root_nid.unsqueeze(1)
            index = index.expand(root_nid.shape[0], feat_size)
            grad_out.scatter_(0, index, grad_var.expand(index.shape[0], feat_size))
        else:
            grad_out = torch.zeros([size, feat_size], dtype=grad_var.dtype)
            scatter_floats(grad_var, root_nid, grad_out)

        if LOG:
            print("var scatter(bck) time: ", time.time() - tic)
        
        return grad_out, None, None, None


def bn1d_var(X, mean, root_nid, X_gat):
    return dist_bn_var.apply(X, mean, root_nid, X_gat)


class local_gather_emb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, root_nid):
        #tic = time.time()
        if py:
            print("Gather code: Python")
            root_nid_index = root_nid.unsqueeze(1).expand(root_nid.shape[0], X.shape[1])
            X_gat = torch.gather(X, 0, root_nid_index)
        else:
            X_gat = torch.empty([root_nid.shape[0], X.shape[1]], dtype=X.dtype)
            gather_floats(X, root_nid, X_gat)

        return X_gat

    @staticmethod
    def backward(ctx, grad):
        return grad, None
    

def gather_emb(X, root_nid):
    return local_gather_emb.apply(X, root_nid)
    

"""
class dist_bn_mean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, root_nid, rank):
        #tic = time.time()
        if py:
            print("Gather code: Python")
            root_nid_index = root_nid.unsqueeze(1).expand(root_nid.shape[0], X.shape[1])
            X_gat = torch.gather(X, 0, root_nid_index)
        else:
            X_gat = torch.empty([root_nid.shape[0], X.shape[1]], dtype=X.dtype)
            gather_floats(X, root_nid, X_gat)
            
        send_n = torch.tensor(X_gat.shape[0], dtype=torch.int64)
        sum_send = torch.sum(X_gat, axis=0)
        
        ## Allreduce 1
        req = dist.all_reduce(sum_send, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        req = dist.all_reduce(send_n, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        mean = sum_send / send_n
        
        ctx.backward_cache = X.shape[0], send_n
        ctx.save_for_backward(root_nid)
        
        return mean

    @staticmethod
    def backward(ctx, grad_mean):    
        size, send_n = ctx.backward_cache
        root_nid, = ctx.saved_tensors
        feat_size = grad_mean.shape[0]

        tic = time.time()        
        req = dist.all_reduce(grad_mean, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        grad_mean /= send_n
        #tic = time.time()
        if py:
            grad_out = torch.zeros([size, feat_size], dtype=grad_mean.dtype)
            index = root_nid.unsqueeze(1)
            index = index.expand(root_nid.shape[0], feat_size)
            grad_out.scatter_(0, index, grad_mean.expand(index.shape[0], feat_size))
        else:
            grad_out = torch.zeros([size, feat_size], dtype=grad_mean.dtype)
            broadcast_floats(grad_mean, root_nid, grad_out)
        print("mean scatter(bck) time: ", time.time() - tic)
        return grad_out, None, None


def bn1d_mean(X, root_nid, rank):
    return dist_bn_mean.apply(X, root_nid, rank)


class dist_bn_var(torch.autograd.Function):
    @staticmethod
    #def forward(ctx, X, mean, send_n, X_gat, root_nid):
    def forward(ctx, X, mean, root_nid):
        #tic = time.time()
        if py:
            root_nid_index = root_nid.unsqueeze(1).expand(root_nid.shape[0], X.shape[1])
            X_gat = torch.gather(X, 0, root_nid_index)
        else:
            X_gat = torch.empty([root_nid.shape[0], X.shape[1]], dtype=X.dtype)
            gather_floats(X, root_nid, X_gat)        
        #print("var gather time: ", time.time() - tic)
        
        # sum_send = torch.sum(X_gat, axis=0)
        send_n = torch.tensor(X_gat.shape[0], dtype=torch.int64)

        ## Allreduce 1
        # req = dist.all_reduce(sum_send, op=dist.ReduceOp.SUM, async_op=True)
        # req.wait()
        req = dist.all_reduce(send_n, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        # mean = sum_send / send_n
        
        var_send = torch.sum((X_gat - mean)**2, axis=0)
        req = dist.all_reduce(var_send, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        variance = var_send / send_n

        ctx.backward_cache = X.shape[0], send_n
        ctx.save_for_backward(X_gat, root_nid)
        return variance

    @staticmethod
    def backward(ctx, grad_var):    
        size, send_n = ctx.backward_cache
        X_gat, root_nid = ctx.saved_tensors
        feat_size = grad_var.shape[0]
        #tic = time.time()
        
        req = dist.all_reduce(grad_var, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        grad_var = grad_var.unsqueeze(0) * 2 * X_gat / send_n
        tic = time.time()
        if py:
            grad_out = torch.zeros([size, feat_size], dtype=grad_var.dtype)
            index = root_nid.unsqueeze(1)
            index = index.expand(root_nid.shape[0], feat_size)
            grad_out.scatter_(0, index, grad_var.expand(index.shape[0], feat_size))
        else:
            grad_out = torch.zeros([size, feat_size], dtype=grad_var.dtype)
            #print("grad_var size: ", grad_var.size(), flush=True)
            #print("root_nid size: ", root_nid.size(), flush=True)        
            scatter_floats(grad_var, root_nid, grad_out)
        
        print("var scatter(bck) time: ", time.time() - tic)
        
        #return grad_out, None, None, None, None
        return grad_out, None, None


def bn1d_var(X, mean, root_nid):
    return dist_bn_var.apply(X, mean, root_nid)
"""
