import torch
from loss.loss import OhemCELoss
from model.model import BiSeNet
from torch.utils.data import random_split
from loss.optimizer import Optimizer
from torch.utils.data import DataLoader
from dataset.pre_train_data_load import pretrain_Facemask
from dataset.pre_train_org_data_load import pretrain_org_Facemask
import logging
import torch.distributed as dist
import os.path as osp
from tqdm import tqdm
from loss.iou import iou
respth = '/res'

model = BiSeNet(n_classes=9)
device = torch.device("cuda")
model = model.to(device)
model.load_state_dict(torch.load('res/cp/class9.pth', map_location=torch.device("cuda")))
print(model)


for param in model.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    if "conv_out" or "conv_out16" or "conv_out32" or "ffm" in name:
        param.requires_grad = True


logging.basicConfig(filename='iter.log', level=logging.INFO)
logger = logging.getLogger()


face_data = 'org-label'
ds = pretrain_Facemask(rootpth=face_data)
ds_org = pretrain_org_Facemask(rootpth=face_data)
allds = torch.utils.data.ConcatDataset([ds, ds_org])
n_img_per_gpu = 8
cropsize = [448, 448]



# 검증 데이터셋 비율 정의
val_size = int(len(allds) * 0.2)  # 예를 들어 20%를 검증에 사용

# 무작위로 데이터셋을 검증 및 훈련 데이터셋으로 분할
train_dataset, val_dataset = random_split(allds, [len(allds) - val_size, val_size])



dl = DataLoader(train_dataset,
                batch_size = n_img_per_gpu,
                shuffle = False,
                pin_memory = True,
                drop_last = True)

val_dl = DataLoader(val_dataset,
                batch_size = n_img_per_gpu,
                shuffle = False,
                pin_memory = True,
                drop_last = True)

# for v_im, v_lb in val_dl:
#     for one in v_lb:
#         np_array = one.numpy()
#         plt.imshow(np_array)
#         plt.show()


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

loss_avg = []
accary = []
epoch = 0
mean_iou = 0.0
# CUDA 사용 가능한 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for it in range(max_iter):
    with tqdm(dl, unit="batch") as tepoch:
        for im,lb in tepoch:
            tepoch.set_description(f'Epoch - {epoch}')
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
            # tepoch.set_postfix(loss=f"{loss.item():4f}")
            # print training log message
            loss_avg.append(loss.item())
            loss_mean = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            tepoch.set_postfix(loss=f"{loss_mean:.4f}")
        with tqdm(val_dl) as valtepoch:
            for v_im,v_lb in valtepoch:
                tepoch.set_description(f'Epoch {epoch} val')
                v_im,v_lb = v_im.to(device),v_lb.cpu()
                y_preds = model(v_im)[0]
                iouscore = iou(y_preds,v_lb)
                accary.append(iouscore)
                mean_iou =  sum(accary) / len(accary)
                valtepoch.set_postfix(mean_iou=f"{mean_iou:.4f}")

        msg = ', '.join([
            'it: {it}/{max_it}',
            'lr: {lr:4f}',
            'loss: {loss:.4f}',
            'mean_iou: {mean_iou}',
        ]).format(
            it=it + 1,
            max_it=max_iter,
            lr=lr,
            loss=loss_mean,
            mean_iou=mean_iou,
        )
        logger.info(msg)
        loss_avg = []
        accary = []
        epoch += 1


        state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state, './res/cp/{}conv_new.pth'.format(it))

#  dump the final model
save_pth = osp.join(respth, 'model_final_diss.pth')
# net.cpu()
state = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
if dist.get_rank() == 0:
    torch.save(state, save_pth)
logger.info('training done, model saved to: {}'.format(save_pth))