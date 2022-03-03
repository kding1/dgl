## Step I: Partitions
### Reddit partition - create 2 partitions
python partition_graph.py --dataset reddit --num-part 2

Note: the partition folder (Libra_result_reddit) is created in the current directory.


## Step II: Distributed runs
Following are the command-line for GraphSAGE model for reddit, ogbn-products, and ogbn-papers100M datasets


cd dgl/examples/pytorch/graphsage/distgnn/
nodes=2
## reddit
exec_file=train_dist_sym.py
dataset=reddit
nepochs=20
lr=0.01
sh run_dist.sh -n $nodes -ppn 2  python $exec_file --dataset $dataset \
   --n-epochs $nepochs \
      --lr  $lr \
         --dropout 0.50 \
            --aggregator-type gcn




## ogbn-products
exec_file=train_dist_sym_ogbn-products.py
dataset=ogbn-products
nepochs=20
lr=0.01
sh run_dist.sh -n $nodes -ppn 2  python $exec_file --dataset $dataset \
   --n-epochs $nepochs \
      --lr  $lr \
         --dropout 0.50 \
            --aggregator-type mean



## ogbn-papers100M
exec_file=train_dist_sym_ogbn-papers.py
dataset=ogbn-papers100M
nepochs=20
lr=0.01
sh run_dist.sh -n $nodes -ppn 2  python $exec_file --dataset $dataset \
   --n-epochs $nepochs \
      --lr  $lr \
         --dropout 0.50 \
            --aggregator-type mean

Note: the distributed runs assume the partition folder is present in current directory.


## Dataset supported
All standard dataset supported by DGL are supported
- Reddit, Cora, Pubmed, Citeseer, ogb datasets
