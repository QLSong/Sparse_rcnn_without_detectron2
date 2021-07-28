export PATH=/opt/conda/bin:$PATH
pip install pycocotools pillow tensorboard matplotlib scipy fvcore -i https://pypi.tuna.tsinghua.edu.cn/simple/
cd /workspace/mnt/storage/songqinglong/code/project/Sparse_rcnn_without_detectron2 && \
python projects/SparseRCNN/train.py --num-gpus 8 --cfg projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml
