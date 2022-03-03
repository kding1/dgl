## Step I: Partitions
### Reddit partition - create 2 partitions
python partition_graph.py --dataset reddit --num-part 2

Note: the partition folder (Libra_result_reddit) is created in the current directory.


## Step II: Distributed runs
run_script.sh contains exmaple command line for reddit, ogbn-products, and ogbn-papers100M datasets

Note: the distributed runs assume the partition folder is present in current directory.

## Dataset supported
All standard dataset supported by DGL are supported
- Reddit, Cora, Pubmed, Citeseer, ogb datasets