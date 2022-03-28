#-*-coding:utf-8-*-

import sys
import os
import yaml
import pprint
import torch
import torch.optim as optim
import torch.distributed as dist
from tqdm import tqdm
from model.superpoint_bn import SuperPointBNNet
from utils.dist import init_distributed_mode, cleanup, reduce_value, is_main_process
from solver.loss import loss_func
from dataset.coco import COCODataset
from torch.utils.data import DataLoader
from utils.log import Log

os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'


def build_data_loader(config, is_train, device='cpu'):
    coco = COCODataset(config['data'], is_train=is_train, device=device)
    # dist sampler
    data_sampler = torch.utils.data.distributed.DistributedSampler(coco)
    b_size = config['solver']['train_batch_size'] if is_train else config['solver']['test_batch_size']
    data_loader = DataLoader(coco,
                             sampler = data_sampler,
                             batch_size=b_size,
                             collate_fn=coco.batch_collator)
    return data_loader, data_sampler


def main(config):
    log = None
    if is_main_process():
        log = Log(config['solver']['save_dir']).run()
        log.info('start training...')
        log.info('{}'.format(pprint.pformat(config)))
    #
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #
    init_distributed_mode(config)#update config,set rank,world_size,local_rank...
    device = torch.device(device)

    # datasets
    train_loader, train_sampler = build_data_loader(config, is_train=True, device=device)
    test_loader, test_sampler = build_data_loader(config, is_train=False, device=device)

    # learning rate
    base_lr = config['solver']['base_lr']*config['solver']['world_size']#学习率要根据并行GPU的数量进行倍增
    if config['solver']['rank'] == 0:  #pring only in the first processer
        if os.path.exists(config['solver']['save_dir']) is False:
            os.makedirs(config['solver']['save_dir'])

    model = SuperPointBNNet(config['model'], device=device,using_bn=True)
    model.to(device)

    # load pretrained weight
    if os.path.exists(config['model']['pretrained_model']):#
        print('Load Pre-Trained Model...')
        pretrained_dict = torch.load(config['model']['pretrained_model'], map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if \
                           k in model_dict and model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(pretrained_dict, strict=False)
    else:# 如果不存在预训练权重，需要将第一个进程中的权重保存，然后其他进程载入，保持初始化权重一致
        checkpoint_path = os.path.join(config['solver']['save_dir'], "auto_initial_weights.pth")
        if config['solver']['rank'] == 0:
            torch.save(model.state_dict(), checkpoint_path)
        dist.barrier()
        # 这里注意，一定要指定map_location参数，否则会导致第一块GPU占用更多资源
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    # 是否同步bn层
    if config['solver']['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)

    # # 是否冻结权重
    # if config['solver']['freeze_backbone']:
    #     for name, para in model.named_parameters():
    #         if 'backbone' in name or 'detector' in name:
    #             para.requires_grad_(False)
    #     params = [p for p in model.parameters() if p.requires_grad]
    # else:# 为不同层设置不同的学习率
    #     ignored_params = list(map(id, model.backbone.parameters()))  # return parameter address
    #     base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    #     params = [{'params': base_params},
    #               {'params': model.backbone.parameters(), 'lr': base_lr / 10}]

    # 转为DDP模型
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config['solver']['gpu']])

    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    #scheduler = lr_scheduler.StepLR(optimizer, 1, gamma=0.6)

    # start train
    for epoch in range(config['solver']['epoch']):
        train_sampler.set_epoch(epoch)

        ##scheduler.step(epoch)
        train_loss = train_one_epoch(config=config,
                                     model=model,
                                     optimizer=optimizer,
                                     data_loader=train_loader,
                                     device=device,
                                     epoch=epoch)
        test_loss = evaluate(config=config,
                             model=model,
                             data_loader=test_loader,
                             device=device)
        test_loss = test_loss / test_sampler.total_size

        if config['solver']['rank'] == 0:
            torch.save(model.module.state_dict(),
                       os.path.join(config['solver']['save_dir'],
                                    config['solver']['model_name']+'_{}_{}.pth').format(epoch, round(test_loss,5)))
            log.info("[save model, epoch: {}] train loss: {} test loss: {}"
                            .format(epoch, round(train_loss, 5), round(test_loss, 5)))

    cleanup()


def train_one_epoch(config, model, optimizer, data_loader, device, epoch):
    model.train()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    # 在进程0中打印训练进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        ##
        if step>1000:
            break
        raw_outputs = model(data['raw'])
        warp_outputs = model(data['warp'])
        prob, desc, prob_warp, desc_warp = raw_outputs['det_info'], \
                                           raw_outputs['desc_info'], \
                                           warp_outputs['det_info'], \
                                           warp_outputs['desc_info']

        loss = loss_func(config['solver'], data, prob, desc,
                         prob_warp, desc_warp, device)

        loss.backward()

        loss = reduce_value(loss, average=True)
        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)#第step迭代时loss均值


        # 在进程0中打印平均loss
        if is_main_process():

            if config['solver']['freeze_backbone']:
                data_loader.desc = "[epoch: {}, lr: {:.2e}] mean loss: {}" \
                    .format(epoch,
                            optimizer.param_groups[0]["lr"],
                            round(mean_loss.item(), 5))
            else:
                data_loader.desc = "[epoch: {}, lr: {:.2e}] mean loss: {}"\
                    .format(epoch,
                            optimizer.param_groups[0]["lr"],
                            #optimizer.param_groups[1]["lr"],
                            round(mean_loss.item(), 5))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    return mean_loss.item()

@torch.no_grad()
def evaluate(config, model, data_loader, device):
    model.eval()

    sum_loss = torch.zeros(1).to(device)

    # 在进程0中打印验证进度
    if is_main_process():
        data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        if step>1000:
            break

        raw_outputs = model(data['raw'])
        warp_outputs = model(data['warp'])
        prob, desc, prob_warp, desc_warp = raw_outputs['det_info'], \
                                           raw_outputs['desc_info'], \
                                           warp_outputs['det_info'], \
                                           warp_outputs['desc_info']

        loss = loss_func(config['solver'], data, prob, desc,
                         prob_warp, desc_warp, device)
        sum_loss += loss

    # 等待所有进程计算完毕
    if device != torch.device("cpu"):
        torch.cuda.synchronize(device)

    reduce_loss = reduce_value(sum_loss, average=False)

    return reduce_loss.item()


if __name__=='__main__':
    # python -m torch.distributed.launch --nproc_per_node=3 --use_env train_dist.py
    config = {}
    with open('./config/superpoint_train.yaml','r') as fin:
        config = yaml.safe_load(fin)
    torch.cuda.empty_cache()
    main(config)
