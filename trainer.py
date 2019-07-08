import torch
from tqdm import tqdm
import numpy as np
import logging 
from evaluate import eval_func, re_rank
from evaluate import euclidean_dist
from utils import AvgerageMeter
import os.path as osp
import os
try:
    from apex import amp
except:
    pass


class BaseTrainer(object):
    def __init__(self, cfg, model, train_dl, val_dl, 
                 optimizer, scheduler, loss_func, num_query):
        self.cfg = cfg
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_iter = iter(train_dl)
        self.val_iter = iter(val_dl)
        self.optim = optimizer
        self.scheduler = scheduler
        self.scheduler.step()
        self.loss_func = loss_func
        self.num_query = num_query

        self.loss_avg = AvgerageMeter()
        self.acc_avg = AvgerageMeter()
        self.train_epoch = 1
        self.batch_cnt = 0

        self.logger = logging.getLogger('reid_baseline.train')
        self.log_period = cfg.SOLVER.LOG_PERIOD
        self.checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
        self.eval_period = cfg.SOLVER.EVAL_PERIOD
        self.output_dir = cfg.OUTPUT_DIR
        self.device = cfg.MODEL.DEVICE
        self.epochs = cfg.SOLVER.MAX_EPOCHS

        self.model.to(self.device)

        self.logger.info('Trainer Built')

        if cfg.SOLVER.FP16:
            try:
                self.model, self.optim = amp.initialize(self.model, self.optim,
                                                        opt_level='O1')
                self.mix_precision = True
                self.logger.info('Using fp16 training')
            except:
                self.mix_precision = False
                self.logger.info('apex not installed, using fp32 training'
                      'install help: https://github.com/NVIDIA/apex/issue/259')


    def _get_train_data(self):
        try:
            batch = next(self.train_iter)
            self.batch_cnt += 1
            self._handle_new_batch()
        except StopIteration:
            self.train_iter = iter(self.train_dl)
            batch = next(self.train_iter)
            self._handle_new_epoch()
        return batch

    def handle_new_batch(self):
        self.batch_cnt += 1
        if self.batch_cnt % self.cfg.SOLVER.LOG_PERIOD == 0:
            self.logger.info('Epoch[{}] Iteration[{}/{}] Loss: {:.3f},'
                            'Acc: {:.3f}, Base Lr: {:.2e}'
                            .format(self.train_epoch, self.batch_cnt,
                                    len(self.train_dl), self.loss_avg.avg,
                                    self.acc_avg.avg, self.scheduler.get_lr()[0]))


    def handle_new_epoch(self):
        self.batch_cnt = 1
        self.scheduler.step()
        self.logger.info('Epoch {} done'.format(self.train_epoch))
        self.logger.info('-' * 20)
        if self.train_epoch % self.checkpoint_period == 0:
            self.save()
        if self.train_epoch % self.eval_period == 0:
            self.evaluate()
        self.train_epoch += 1

    def step(self, batch):
        self.model.train()
        self.optim.zero_grad()
        img, target = batch
        img, target = img.cuda(), target.cuda()
        score, feat = self.model(img)
        loss = self.loss_func(score, feat, target)
        if self.mix_precision:
            with amp.scale_loss(loss, self.optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optim.step()

        acc = (score.max(1)[1] == target).float().mean()

        self.loss_avg.update(loss.cpu().item())
        self.acc_avg.update(acc.cpu().item())
        
        return self.loss_avg.avg, self.acc_avg.avg

    def evaluate(self):
        self.model.eval()
        num_query = self.num_query
        feats, pids, camids = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.val_dl, total=len(self.val_dl),
                             leave=False):
                data, pid, camid, _ = batch
                data = data.cuda()
                feat = self.model(data).detach().cpu()
                feats.append(feat)
                pids.append(pid)
                camids.append(camid)
        feats = torch.cat(feats, dim=0)
        pids = torch.cat(pids, dim=0)
        camids = torch.cat(camids, dim=0)

        query_feat = feats[:num_query]
        query_pid = pids[:num_query]
        query_camid = camids[:num_query]

        gallery_feat = feats[num_query:]
        gallery_pid = pids[num_query:]
        gallery_camid = camids[num_query:]
        
        distmat = euclidean_dist(query_feat, gallery_feat)

        cmc, mAP, _ = eval_func(distmat.numpy(), query_pid.numpy(), gallery_pid.numpy(), 
                             query_camid.numpy(), gallery_camid.numpy(),
                             use_cython=True)
        self.logger.info('Validation Result:')
        for r in self.cfg.TEST.CMC:
            self.logger.info('CMC Rank-{}: {:.2%}'.format(r, cmc[r-1]))
        self.logger.info('mAP: {:.2%}'.format(mAP))
        self.logger.info('-' * 20)

    def save(self):
        torch.save(self.model.state_dict(), osp.join(self.output_dir,
                self.cfg.MODEL.NAME + '_epoch' + str(self.train_epoch) + '.pth'))
        torch.save(self.optim.state_dict(), osp.join(self.output_dir,
                self.cfg.MODEL.NAME + '_epoch'+ str(self.train_epoch) + '_optim.pth'))



