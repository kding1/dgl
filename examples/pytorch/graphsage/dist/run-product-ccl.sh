
python3 ../../../../tools/launch.py \
	--workspace /home/kding1/dgl/examples/pytorch/graphsage/dist \
	--num_trainers 2 \
	--num_samplers 0 \
	--num_servers 1 \
	--part_config data/ogb-product.json \
	--ip_config ip_config_hdr.txt \
	"python3 train_dist.py --graph_name ogb-product --ip_config ip_config_hdr.txt --num_epochs 30 --batch_size 1000 --backend ccl"
