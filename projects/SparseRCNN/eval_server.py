from __future__ import print_function

import json
import os
import sys
import time
import shutil

from multiprocessing import Process, Queue
from collections import namedtuple

from flask import Flask
from flask import request

from train import eval
from sparsercnn.config import _C as cfg
from sparsercnn.dataset.coco import CocoDataset
from sparsercnn.detector import SparseRCNN
from sparsercnn.evaluation.coco_evaluation import COCOEvaluator
import sparsercnn.dataset.transforms as T
from sparsercnn.dataloader.collate import Collate
import torch

import logging

yaml = 'projects/SparseRCNN/configs/sparsercnn.res50.100pro.3x.yaml'
cfg.merge_from_file(yaml)
device = torch.device('cuda')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
log_path = cfg.OUTPUT_DIR
log_name = log_path+'/eval_'+rq+'.log'
log_file = log_name
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file, mode='w')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
logger.addHandler(ch)
logger.addHandler(fh)

test_transforms = T.Compose([
    T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING),
])
test_dataset = CocoDataset(cfg, 'val', test_transforms)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=Collate(cfg)
)

EvalTask = namedtuple('EvalTask', ['val_root', 'checkpoint'])


class AsyncEval(object):
    def __init__(self):
        self.queue = Queue()
        self.exit = False
        self.worker = Process(target=self.loop, args=())
        self.worker.daemon = True
        self.worker.start()

    def add_task(self, val_root, checkpoint):
        self.queue.put(EvalTask(val_root=val_root, checkpoint=checkpoint))

    def close(self):
        self.queue.put(None)
        self.worker.join()

    def loop(self):
        while True:
            task = self.queue.get()
            if not task:
                break
            run(task)


best_criterion = 0

def run(task):
    model = SparseRCNN(cfg)
    logger.info('load checkpoint : '+task.checkpoint)

    state_dict = torch.load(task.checkpoint, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict['state_dict'].items():
        new_state_dict[k[7:] if k.startswith('module.') else k] = v
    model.load_state_dict(new_state_dict)
    model.cuda()

    evaluator = COCOEvaluator(cfg.BASE_ROOT, cfg.DATASETS.TEST[0], logger)

    eval(model, device, test_loader, logger, evaluator)


validater = AsyncEval()

app = Flask(__name__)

@app.route('/' , methods = ['POST'])
def index():
    request_data = json.loads(request.get_data())
    print(request_data)
    if 'checkpoint' in request_data:
        global validater
        validater.add_task(request_data['val_root'], request_data['checkpoint'])
        response = {'message': 'Task add successful'}
    else:
        response = {'message': 'Wrong request data'}

    return json.dumps(response)


if __name__ == '__main__':

    app.run(debug = True, host = '0.0.0.0', port = 10111)
