import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import uniform
import torch.distributed as dist

class dist_batch_norm(nn.Module):

    def __init__(self, in_size, root_nid, momentum=0.9, eps = 1e-5):
        super(dist_batch_norm, self).__init__()

        self.root_nid = root_nid
        self.momentum = momentum
        self.insize = in_size
        self.eps = eps
        
        U = uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        self.gamma = nn.Parameter(U.sample(torch.Size([self.insize])).view(self.insize))
        self.beta = nn.Parameter(torch.zeros(self.insize))            

    def forward(self, input):
        
        X = input
        #print("gamma: {}, beta: {}".format(self.gamma, self.beta))
        #mean = torch.mean(X, axis=0)
        #variance = torch.mean((X-mean)**2, axis=0)
        #num_feats = X.shape[1]
        #X_gat = torch.gather()
        mean = bn1d_mean(X, self.root_nid)
        #mean = torch.zeros(X.shape[1])
        ##variance = bn1d_var(X, self.root_nid)
        variance = torch.ones(X.shape[1])
        
        X_hat = (X-mean) * 1.0 /torch.sqrt(variance + self.eps)
        out = self.gamma * X_hat + self.beta
          
        #return out
        return input


class dist_bn_mean(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, root_nid):
        root_nid_index = root_nid.unsqueeze(1).expand(root_nid.shape[0], X.shape[1])
        X_gat = torch.gather(X, 0, root_nid_index)
        
        sum_send = torch.sum(X_gat, axis=0)
        send_n = torch.tensor(X_gat.shape[0], dtype=torch.int64)
        
        print("root_nid_index: ", root_nid_index)
        print("sum_send: ", sum_send)
        print("#nodes in part: ", send_n)
        ## Allreduce 1
        req = dist.all_reduce(sum_send, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        ##req = dist.all_reduce(send_n, op=dist.ReduceOp.SUM, async_op=True)
        ##req.wait()

        mean = sum_send / send_n
        
        #var_send = torch.sum((X_gat - mean)**2, axis=0)
        #req = dist.all_reduce(var_send, op=dist.ReduceOp.SUM, async_op=True)
        #req.wait()
        #variance = var_send / send_n
        print("mean: ", mean)
        #print("variance: ", variance)
        print("n: ", send_n)        

        ctx.backward_cache = X.shape[0], send_n
        ctx.save_for_backward(root_nid)
        
        return mean

    @staticmethod
    def backward(ctx, grad_mean):    
        size, send_n = ctx.backward_cache
        root_nid, = ctx.saved_tensors
        feat_size = grad_mean.shape[0]

        grad_out = torch.zeros([size, feat_size], dtype=grad_mean.dtype)
        return grad_out, None
        
        req = dist.all_reduce(grad_mean, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        grad_mean /= send_n
        grad_out = torch.zeros([size, feat_size], dtype=grad_mean.dtype)
        index = root_nid.unsqueeze(1)
        index = index.expand(root_nid.shape[0], feat_size)
        grad_out.scatter_(0, index, grad_mean.expand(index.shape[0], feat_size))
        return grad_out, None


def bn1d_mean(X, root_nid):
    return dist_bn_mean.apply(X, root_nid)


class dist_bn_var(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, root_nid):
        print("root_nid: ", root_nid)
        print("root_nid shape: ", root_nid.size())
        root_nid_index = root_nid.unsqueeze(1).expand(root_nid.shape[0], X.shape[1])
        X_gat = torch.gather(X, 0, root_nid_index)

        sum_send = torch.sum(X_gat, axis=0)
        send_n = torch.tensor(X_gat.shape[0], dtype=torch.int64)

        ## Allreduce 1
        req = dist.all_reduce(sum_send, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        req = dist.all_reduce(send_n, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        mean = sum_send / send_n
        
        var_send = torch.sum((X_gat - mean)**2, axis=0)
        req = dist.all_reduce(var_send, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()

        variance = var_send / send_n
        #print("mean: ", mean)
        print("variance: ", variance)
        print("n: ", send_n)        

        ctx.backward_cache = X.shape[0], send_n
        ctx.save_for_backward(X_gat, root_nid)
        return variance

    @staticmethod
    def backward(ctx, grad_var):    
        size, send_n = ctx.backward_cache
        X_gat, root_nid = ctx.saved_tensors
        feat_size = grad_var.shape[0]

        grad_out = torch.zeros([size, feat_size], dtype=grad_var.dtype)
        return grad_out, None
        
        req = dist.all_reduce(grad_var, op=dist.ReduceOp.SUM, async_op=True)
        req.wait()
        grad_var = grad_var.unsqueeze(0) * 2 * X_gat / send_n

        grad_out = torch.zeros([size, feat_size], dtype=grad_var.dtype)
        index = root_nid.unsqueeze(1)
        index = index.expand(root_nid.shape[0], feat_size)
        grad_out.scatter_(0, index, grad_var.expand(index.shape[0], feat_size))
        return grad_out, None


def bn1d_var(X, root_nid):
    return dist_bn_var.apply(X, root_nid)
