import torch
from loss.loss import OhemCELoss
from model.model import BiSeNet
from model.model import BiSeNetOutput
from loss.optimizer import Optimizer
from torch.utils.data import DataLoader
import time
from dataset.pre_train_data_load import pretrain_Facemask
import datetime
import logging
import torch.distributed as dist
import os.path as osp
from tqdm import tqdm
respth = '/res'

model = BiSeNet(n_classes=19)
device = torch.device("cuda")
model = model.to(device)
model.load_state_dict(torch.load('res/cp/org_iter.pth', map_location=torch.device("cuda")))
print(model)
for param in model.parameters():
    param.requires_grad = False

logger = logging.getLogger()

num_classes= 9
model.conv_out = BiSeNetOutput(256, 256, num_classes).to(device)
model.conv_out16 = BiSeNetOutput(128, 64, num_classes).to(device)
model.conv_out32 = BiSeNetOutput(128, 64, num_classes).to(device)
print(model)

face_data = 'org-label'
ds = pretrain_Facemask(rootpth=face_data)
n_img_per_gpu = 8
cropsize = [448, 448]
dl = DataLoader(ds,
                batch_size = n_img_per_gpu,
                shuffle = False,
                pin_memory = True,
                drop_last = True,
                )


momentum = 0.9
weight_decay = 5e-4
lr_start = 1e-2
max_iter = 80000
power = 0.9
warmup_steps = 1000
warmup_start_lr = 1e-5
score_thres = 0.7
ignore_idx = -100
n_min = n_img_per_gpu * cropsize[0] * cropsize[1]//16

LossP = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
Loss2 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)
Loss3 = OhemCELoss(thresh=score_thres, n_min=n_min, ignore_lb=ignore_idx)

optim = Optimizer(
        model =model,
        lr0 = lr_start,
        momentum = momentum,
        wd = weight_decay,
        warmup_steps = warmup_steps,
        warmup_start_lr = warmup_start_lr,
        max_iter = max_iter,
        power = power)

msg_iter = 50
loss_avg = []
st = glob_st = time.time()
diter = iter(dl)
epoch = 0

# CUDA 사용 가능한 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pbar = tqdm(total=max_iter, desc='Training Progress', unit='epoch')

for it in range(max_iter):
    pbar.update(1)
    try:
        im, lb = next(diter)
        if not im.size()[0] == n_img_per_gpu:
            raise StopIteration
    except StopIteration:
        epoch += 1
        diter = iter(dl)
        im, lb = next(diter)

    im = im.to(device)
    lb = lb.to(device)
    H, W = im.size()[2:]
    lb = torch.squeeze(lb, 1)

    optim.zero_grad()
    out, out16, out32 = model(im)

    lossp = LossP(out, lb)
    loss2 = Loss2(out16, lb)
    loss3 = Loss3(out32, lb)
    loss = lossp + loss2 + loss3
    loss.backward()
    optim.step()

    loss_avg.append(loss.item())

    #  print training log message
    if (it+1) % msg_iter == 0:
        loss_avg = sum(loss_avg) / len(loss_avg)
        lr = optim.lr
        ed = time.time()
        t_intv, glob_t_intv = ed - st, ed - glob_st
        eta = int((max_iter - it) * (glob_t_intv / it))
        eta = str(datetime.timedelta(seconds=eta))
        msg = ', '.join([
                'it: {it}/{max_it}',
                'lr: {lr:4f}',
                'loss: {loss:.4f}',
                'eta: {eta}',
                'time: {time:.4f}',
            ]).format(
                it = it+1,
                max_it = max_iter,
                lr = lr,
                loss = loss_avg,
                time = t_intv,
                eta = eta
            )
        logger.info(msg)
        pbar.set_postfix(loss=loss_avg)
        print(msg)
        loss_avg = []
        st = ed
        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state, './res/cp/{}_iter.pth'.format(it))

#  dump the final model
save_pth = osp.join(respth, 'model_final_diss.pth')
# net.cpu()
state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
if dist.get_rank() == 0:
    torch.save(state, save_pth)
logger.info('training done, model saved to: {}'.format(save_pth))