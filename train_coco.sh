# python projects/SparseRCNN/train.py --num-gpus 4 --cfg projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml
python projects/SparseRCNN/train_dist.py --num-gpus 4 --cfg projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml --world-size 1 \
--dist-url "tcp://localhost:10001" --rank 0 --multiprocessing-distributed 
