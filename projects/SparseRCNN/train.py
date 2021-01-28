from sparsercnn.config import _C as cfg

from sparsercnn import add_sparsercnn_config
from sparsercnn.dataset.coco import CocoDataset
from sparsercnn.dataset.voc import VOCDataset
from sparsercnn.detector import SparseRCNN, Loss
from sparsercnn.solver.build_optimizer import *
from sparsercnn.evaluation.voc_evaluation import PascalVOCDetectionEvaluator
from sparsercnn.evaluation.coco_evaluation import COCOEvaluator
import torch
import torch.optim as optim
from sparsercnn.dataloader.collate import Collate
import argparse
import sparsercnn.dataset.transforms as T
import os
import logging
import time
from sparsercnn.dataloader.sampler import AspectRatioBasedSampler, RandomSampler
from sparsercnn.dataloader.data_parallel import DataParallel


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help = 'experiment configure file name',
                        required = True,
                        type = str)
    parser.add_argument('--num-gpus',
                        help = 'num of gpus',
                        type = int,
                        default = 2)
    parser.add_argument('--eval-only',
                        help = 'only evalution',
                        action='store_true')
    parser.add_argument('--weights',
                        help = 'model root',
                        type = str,
                        default='')
    parser.add_argument('--standard-size',
                        help = 'if true, every gpu has same batch',
                        action = 'store_true')
    parser.add_argument('--batch-difference',
                        help = 'first gpus batch add this value equal others batch',
                        type = int,
                        default = 1)

    args = parser.parse_args()

    return args

def setup(args):
    """
    Create configs and perform basic setups.
    """
    args = parse_args()    
    cfg.merge_from_file(args.cfg)
    # cfg.freeze()

    return cfg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

def eval(model, device, test_dataloader, logger, evaluator):
    model.eval()
    
    for idx, (image, img_whwh, target) in enumerate(test_dataloader):
        image = image.to(device)
        img_whwh = img_whwh.to(device)
    
        y = model(image, img_whwh)

        for t in target:
            for k in t.keys():
                if k in ['gt_boxes', 'gt_classes', 'image_size_xyxy', 'image_size_xyxy_tgt']:
                    t[k] = t[k].to(device)

        for j in range(y[1].shape[0]):
            _t = target[j]
            y[1][j] *= torch.Tensor([_t['width'], _t['height'], _t['width'], _t['height']]).to(device)/_t['image_size_xyxy'] 

        evaluator.process(target, y)
        if idx % 100 == 0:
            msg = 'Evaluation: [{0}/{1}]'.format(
                      idx, len(test_dataloader),
                  )
            logger.info(msg)
        # break

    ret = evaluator.evaluate()
    # print(ret)
    logger.info('AP : {AP:.3f}, AP50 : {AP50:.3f}, AP75 : {AP75:.3f}'.format(
                    AP=ret["bbox"]["AP"],
                    AP50=ret["bbox"]["AP50"],
                    AP75=ret["bbox"]["AP75"]))
    return 

def train_one_epoch(epoch, model, device, criterion, train_dataloader, optimizer, lr_scheduler, logger):
    model.train()
    class_weight = cfg.MODEL.SparseRCNN.CLASS_WEIGHT
    giou_weight = cfg.MODEL.SparseRCNN.GIOU_WEIGHT
    l1_weight = cfg.MODEL.SparseRCNN.L1_WEIGHT
    weight_dict = {"loss_ce": class_weight, "loss_bbox": l1_weight, "loss_giou": giou_weight}

    losses_ce = AverageMeter()
    losses_bbox = AverageMeter()
    losses_giou = AverageMeter()
    total_losses = AverageMeter()

    for i, (image, img_whwh, target) in enumerate(train_dataloader):

        image = image.to(device)
        img_whwh = img_whwh.to(device)
        y = model(image, img_whwh)

        for t in target:
            for k in t.keys():
                if k in ['gt_boxes', 'gt_classes', 'image_size_xyxy', 'image_size_xyxy_tgt']:
                    t[k] = t[k].to(device)

        loss = criterion(y, target)
        
        total_loss = 0
        loss_ce = 0
        loss_bbox = 0
        loss_giou = 0
        for k in loss.keys():
            if 'loss_ce' in k:
                loss[k] *= weight_dict['loss_ce']
                loss_ce += loss[k]
            elif 'loss_bbox' in k:
                loss[k] *= weight_dict['loss_bbox']
                loss_bbox += loss[k]
            elif 'loss_giou' in k:
                loss[k] *= weight_dict['loss_giou']
                loss_giou += loss[k]
            total_loss += loss[k]
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        losses_ce.update(loss_ce.cpu().data.numpy())
        losses_bbox.update(loss_bbox.cpu().data.numpy())
        losses_giou.update(loss_giou.cpu().data.numpy())
        total_losses.update(total_loss.cpu().data.numpy())
        if i % cfg.TRAIN_PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Total Loss: {total_losses.avg:.3f}\t' \
                  'Loss_ce: {losses_ce.avg:.3f}\t' \
                  'Loss_bbox: {losses_bbox.avg:.3f}\t' \
                  'Loss_giou: {losses_giou.avg:.3f}'.format(
                      epoch, i, len(train_dataloader),
                      total_losses = total_losses,
                      losses_ce = losses_ce,
                      losses_bbox = losses_bbox,
                      losses_giou = losses_giou
                  )
            logger.info(msg)
        # break

def train(args):
    cfg = setup(args)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = cfg.OUTPUT_DIR
    if args.eval_only:
        log_name = log_path+'/eval_'+rq+'.log'
    else:
        log_name = log_path+'/train_'+rq+'.log'
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
    logger.info(cfg)
    
    min_size = cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    device = torch.device(cfg.MODEL.DEVICE)
    sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    transforms = T.Compose([
        T.RandomFlip(),
        T.ResizeShortestEdge(min_size, max_size, sample_style),
    ])
    test_transforms = T.Compose([
        T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST, sample_style),
    ])

    model = SparseRCNN(cfg)

    start_epoch = 0
    if args.weights != '':
        state_dict = torch.load(args.weights, map_location='cpu')#['model']
        # start_epoch = state_dict['Epoch']
        new_state_dict = {}
        # for k, v in state_dict['state_dict'].items():
        #     new_state_dict[k[7:]] = v
        for k, v in state_dict.items():
            if ('conv' in k or 'fpn' in k or 'shortcut' in k) and 'norm' not in k:
                if 'weight' in k:
                    new_state_dict[k.replace('weight', 'conv2d.weight')] = v
                if 'bias' in k:
                    new_state_dict[k.replace('bias', 'conv2d.bias')] = v
            else:
                new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    if cfg.MODEL.DEVICE == 'cuda':
        model = model.cuda()
        if not args.standard_size and args.num_gpus > 1:
            _gpus = range(args.num_gpus)
            chunk_sizes = [cfg.SOLVER.IMS_PER_BATCH - args.batch_difference]
            rest_batch_size = (cfg.SOLVER.IMS_PER_BATCH*len(_gpus) - chunk_sizes[0])
            for i in range(len(_gpus) - 1):
                slave_chunk_size = rest_batch_size // (len(_gpus) - 1)
                if i < rest_batch_size % (len(_gpus) - 1):
                    slave_chunk_size += 1
                chunk_sizes.append(slave_chunk_size)
            model = DataParallel(model, device_ids=_gpus, chunk_sizes=chunk_sizes).cuda()
        else:
            model = torch.nn.DataParallel(model)

    if cfg.DATASETS.TRAIN[0].startswith('voc'):
        train_dataset = VOCDataset(cfg, 'train', transforms)
        test_dataset = VOCDataset(cfg, 'val', test_transforms)
        train_sampler = RandomSampler(train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH * args.num_gpus, drop_last=True)
        evaluator = PascalVOCDetectionEvaluator(cfg.DATASETS.TEST[0], logger)
    elif cfg.DATASETS.TRAIN[0].startswith('coco'):
        train_dataset = CocoDataset(cfg, 'train', transforms)
        test_dataset = CocoDataset(cfg, 'val', test_transforms)
        train_sampler = AspectRatioBasedSampler(train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH * args.num_gpus, drop_last=True)
        evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], logger)
    else:
        raise('dataset not support!!!')

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        collate_fn=Collate(cfg),
        batch_sampler=train_sampler
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1, #args.num_gpus, 
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        collate_fn=Collate(cfg)
    )

    if args.eval_only:
        eval(model, device, test_loader, logger, evaluator)
        return

    criterion = Loss(cfg).cuda()

    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    start_epoch = 0
    if args.weights != '':
        state_dict = torch.load(args.weights)
        start_epoch = state_dict['Epoch']
        model.load_state_dict(state_dict['state_dict'])
    
    MAXEPOCH = int(cfg.SOLVER.MAX_ITER) // len(train_loader) + 1
    logger.info('MAXEPOCH : %d'%MAXEPOCH)
    for epoch in range(start_epoch, MAXEPOCH):
        train_one_epoch(epoch, model, device, criterion, train_loader, optimizer, lr_scheduler, logger)

        state = {'Epoch' : epoch,
                 'state_dict' : model.state_dict()}
        torch.save(state, cfg.OUTPUT_DIR + '/model_final.pth.tar')
        logger.info('saving models to ' + cfg.OUTPUT_DIR + '/model_final.pth.tar')
        
        if epoch % 10 == 0 and epoch != 0:
            eval(model.module, device, test_loader, logger, evaluator)


if __name__ == '__main__':
    args = parse_args()
    train(args)