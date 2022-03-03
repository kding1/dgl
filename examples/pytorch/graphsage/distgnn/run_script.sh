
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
