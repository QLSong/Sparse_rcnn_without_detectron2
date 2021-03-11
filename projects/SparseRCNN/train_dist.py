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
import torch.multiprocessing as mp
import torch.distributed as dist
from sparsercnn.dataloader.collate import Collate
import argparse
import sparsercnn.dataset.transforms as T
import os
import logging
import time
from sparsercnn.dataloader.sampler import AspectRatioBasedSampler, RandomSampler, DistributedGroupSampler
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
                        default = -1)
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
                        default = 0)
    parser.add_argument('--dist-url', 
                        help ='url used to set up distributed training',
                        type = str,
                        default='tcp://224.66.41.62:23456')
    parser.add_argument('--dist-backend', 
                        help ='distributed backend',
                        type = str,
                        default='nccl')
    parser.add_argument('--world-size', 
                        help='number of nodes for distributed training',
                        type=int,
                        default=-1)
    parser.add_argument('--rank',
                        help='node rank for distributed training',
                        type=int,
                        default=-1)
    parser.add_argument('--multiprocessing-distributed', 
                        help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training',
                         action='store_true')
    parser.add_argument('--gpu', 
                        help='GPU id to use.',
                        type=int,
                        default=None)

    args = parser.parse_args()

    return args

def setup(args):
    """
    Create configs and perform basic setups.
    """
    args = parse_args()    
    cfg.merge_from_file(args.cfg)

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
        y = model(image.float(), img_whwh)

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

    ret = evaluator.evaluate()
    logger.info('AP : {AP:.3f}, AP50 : {AP50:.3f}, AP75 : {AP75:.3f}'.format(
                    AP=ret["bbox"]["AP"],
                    AP50=ret["bbox"]["AP50"],
                    AP75=ret["bbox"]["AP75"]))
    return ret["bbox"]["AP"]

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
        image = image.cuda(device, non_blocking=True)
        img_whwh = img_whwh.cuda(device, non_blocking=True)
        y = model(image, img_whwh)
        for t in target:
            for k in t.keys():
                if k in ['gt_boxes', 'gt_classes', 'image_size_xyxy', 'image_size_xyxy_tgt']:
                    t[k] = t[k].cuda(device)

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
        if (i % cfg.TRAIN_PRINT_FREQ) == 0 and (device == torch.device(0)) :
            msg = 'Epoch: [{0}][{1}/{2}]  ' \
                  'Total Loss: {total_loss:.3f}  '.format(
                      epoch, i, len(train_dataloader),
                      total_loss = total_loss.item(),
                  )
            for k in loss.keys():
                msg += k + ': %.4f  '%(loss[k])
            logger.info(msg)

def get_logger(cfg, args):
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
    # logger.info(cfg)
    return logger

def train(args):
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])
    
    os.environ["WORLD_SIZE"] = str(args.world_size)
    os.environ["RANK"] = str(args.rank)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    if args.num_gpus > 0 :
        ngpus_per_node = args.num_gpus
        args.num_gpus = ngpus_per_node

    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(args.num_gpus, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, args)
    


def main_worker(gpu,gpuNum, args):
    cfg = setup(args)
    args.gpu = gpu
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * args.num_gpus + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    print("init rank ", args.rank, " gpu ", gpu )
    logger = get_logger(cfg, args)

    min_size = cfg.INPUT.MIN_SIZE_TRAIN
    max_size = cfg.INPUT.MAX_SIZE_TRAIN
    device = torch.device(args.gpu)
    sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    transforms = T.Compose([
        T.RandomFlip(),
        T.ResizeShortestEdge(min_size, max_size, sample_style),
    ])
    test_transforms = T.Compose([
        T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST, sample_style),
    ])

    model = SparseRCNN(cfg)
    # from thop import profile
    # x = torch.Tensor(1,3,224,224)
    # wh = torch.Tensor(1,4)
    # macs, params = profile(model, inputs=(x, wh))
    # print(macs, params)
    start_epoch = 0
    if args.weights != '':
        if 'model' not in torch.load(args.weights, map_location='cpu'):
            state_dict = torch.load(args.weights, map_location='cpu')
            new_state_dict = {}
            for k, v in state_dict['state_dict'].items():
                new_state_dict[k[7:]] = v
            start_epoch = state_dict['Epoch']
        else:
            state_dict = torch.load(args.weights, map_location='cpu')['model']
            new_state_dict = {}
            # for k, v in state_dict['state_dict'].items():
            #     new_state_dict[k[7:]] = v
            for k, v in state_dict.items():
                # if ('conv' in k or 'fpn' in k or 'shortcut' in k) and 'norm' not in k:
                # if 'head.head_series.3' in k or 'head.head_series.4' in k or 'head.head_series.5' in k:
                #     continue
                if ('fpn' in k or 'shortcut' in k) and 'norm' not in k:
                    if 'weight' in k:
                        new_state_dict[k.replace('weight', 'conv2d.weight')] = v
                    if 'bias' in k:
                        new_state_dict[k.replace('bias', 'conv2d.bias')] = v
                else:
                    new_state_dict[k] = v
        # state_dict = torch.load(args.weights, map_location='cpu')
        model.load_state_dict(new_state_dict)
    print("load model with rank ", args.rank, " gpu ", device )
        # new_state_dict = {}
        # for k, v in state_dict['state_dict'].items():
        #     new_state_dict[k[7:]] = v
        # # for k, v in state_dict.items():
        # #     # if ('conv' in k or 'fpn' in k or 'shortcut' in k) and 'norm' not in k:
        # #     # if 'head.head_series.3' in k or 'head.head_series.4' in k or 'head.head_series.5' in k:
        # #     #     continue
        # #     if ('fpn' in k or 'shortcut' in k) and 'norm' not in k:
        # #         if 'weight' in k:
        # #             new_state_dict[k.replace('weight', 'conv2d.weight')] = v
        # #         if 'bias' in k:
        # #             new_state_dict[k.replace('bias', 'conv2d.bias')] = v
        # #     else:
        # #         new_state_dict[k] = v
        # model.load_state_dict(new_state_dict)
    if args.multiprocessing_distributed:
        if args.gpu is not None:
            torch.cuda.set_device(device)
            model.cuda(device)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = cfg.SOLVER.IMS_PER_BATCH
            args.workers = args.batch_size
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    else:
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
        if args.distributed:
            train_sampler = DistributedGroupSampler(train_dataset, samples_per_gpu=args.batch_size)
        else:
            train_sampler = AspectRatioBasedSampler(train_dataset, batch_size=cfg.SOLVER.IMS_PER_BATCH * args.num_gpus, drop_last=True)
        evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], logger)
    else:
        raise('dataset not support!!!')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size = args.batch_size,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        collate_fn=Collate(cfg),
        sampler=train_sampler
    )

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, 
    #     num_workers=cfg.DATALOADER.NUM_WORKERS,
    #     pin_memory=True,
    #     collate_fn=Collate(cfg),
    #     batch_sampler=train_sampler
    # )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=1,
        shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        pin_memory=True,
        collate_fn=Collate(cfg)
    )

    if args.eval_only:
        evaluator.reset()
        eval(model.module, device, test_loader, logger, evaluator)
        return

    criterion = Loss(cfg).cuda(device)

    optimizer = build_optimizer(cfg, model)
    lr_scheduler = build_lr_scheduler(cfg, optimizer)
    
    MAXEPOCH = int(cfg.SOLVER.MAX_ITER) // len(train_loader) + 1
    logger.info('MAXEPOCH : %d'%MAXEPOCH)
    max_ap = 0
    for epoch in range(start_epoch, MAXEPOCH):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, device, criterion, train_loader, optimizer, lr_scheduler, logger)

        state = {'Epoch' : epoch,
                 'state_dict' : model.state_dict()}
        torch.save(state, cfg.OUTPUT_DIR + '/model_final_' + str(epoch) + '.pth.tar')
        logger.info('saving models to ' + cfg.OUTPUT_DIR + '/model_final.pth.tar')
        
        # if (epoch + 1) % 1 == 0:
        #     evaluator.reset()
        #     if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % args.num_gpus == 0):
        #         ap = eval(model.module, device, test_loader, logger, evaluator)
        #         if ap > max_ap:
        #             torch.save(state, cfg.OUTPUT_DIR + '/model_best.pth.tar')
        #             max_ap = ap


if __name__ == '__main__':
    args = parse_args()
    train(args)