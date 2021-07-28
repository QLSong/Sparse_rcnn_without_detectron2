export PATH=/opt/conda/bin:$PATH
pip install pycocotools scipy fvcore requests -i https://pypi.tuna.tsinghua.edu.cn/simple/

cd /workspace/mnt/storage/songqinglong/code/project/Sparse_rcnn_without_detectron2
# cp efficientnet-b0-355c32eb.pth /root/.cache/torch/hub/checkpoints/
# python projects/SparseRCNN/train_dist.py --num-gpus 8 --cfg projects/SparseRCNN/configs/sparsercnn.efficientnetb0.100pro.3x.yaml --world-size $WORLD_SIZE \
# --dist-url "tcp://"$MASTER_ADDR":"$MASTER_PORT --rank $RANK --multiprocessing-distributed

# python projects/SparseRCNN/train_dist.py --num-gpus 8 --cfg projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml --world-size $WORLD_SIZE \
# --dist-url "tcp://"$MASTER_ADDR":"$MASTER_PORT --rank $RANK --multiprocessing-distributed

python projects/SparseRCNN/train_dist.py --num-gpus -1 --cfg projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml --world-size 1 \
--dist-url "tcp://localhost:12345" --rank 0 --multiprocessing-distributed