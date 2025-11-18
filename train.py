from MDCANet import Restormer_Encoder_SAR, Restormer_Encoder_Opt, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction_Opt, DetailFeatureExtraction_SAR
from utils.dataset import H5Dataset
import os
import shutil
import sys
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.loss import Fusionloss, cc, TV_Loss, spectral_angle_mapper
import kornia
import torch.nn.parallel
import torch.distributed as dist

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
criteria_fusion = Fusionloss()
best_loss = float('inf')

num_epochs = 120
epoch_gap = 40

lr = 1e-4
weight_decay = 0
batch_size = 8
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
coeff_mse_loss_VF = 1.
coeff_mse_loss_IF = 1.
coeff_decomp = 2.
coeff_tv = 5.

clip_grad_norm_value = 0.01
optim_step = 30
optim_gamma = 1.0

device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIDF_Encoder_Opt = nn.DataParallel(Restormer_Encoder_Opt()).to(device)
DIDF_Encoder_SAR = nn.DataParallel(Restormer_Encoder_SAR()).to(device)
DIDF_Decoder = nn.DataParallel(Restormer_Decoder()).to(device)
BaseFuseLayer = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
DetailFuseLayer_SAR = nn.DataParallel(DetailFeatureExtraction_SAR(num_layers=1)).to(device)
DetailFuseLayer_Opt = nn.DataParallel(DetailFeatureExtraction_Opt(num_layers=1)).to(device)

optimizer1 = torch.optim.Adam(
    DIDF_Encoder_Opt.parameters(), lr=lr, weight_decay=weight_decay)
optimizer5 = torch.optim.Adam(
    DIDF_Encoder_SAR.parameters(), lr=lr, weight_decay=weight_decay)
optimizer2 = torch.optim.Adam(
    DIDF_Decoder.parameters(), lr=lr, weight_decay=weight_decay)
optimizer3 = torch.optim.Adam(
    BaseFuseLayer.parameters(), lr=lr, weight_decay=weight_decay)
optimizer4 = torch.optim.Adam(
    DetailFuseLayer_Opt.parameters(), lr=lr, weight_decay=weight_decay)
optimizer6 = torch.optim.Adam(
    DetailFuseLayer_SAR.parameters(), lr=lr, weight_decay=weight_decay)

scheduler1 = torch.optim.lr_scheduler.StepLR(optimizer1, step_size=optim_step, gamma=optim_gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(optimizer2, step_size=optim_step, gamma=optim_gamma)
scheduler3 = torch.optim.lr_scheduler.StepLR(optimizer3, step_size=optim_step, gamma=optim_gamma)
scheduler4 = torch.optim.lr_scheduler.StepLR(optimizer4, step_size=optim_step, gamma=optim_gamma)
scheduler5 = torch.optim.lr_scheduler.StepLR(optimizer5, step_size=optim_step, gamma=optim_gamma)
scheduler6 = torch.optim.lr_scheduler.StepLR(optimizer6, step_size=optim_step, gamma=optim_gamma)

MSELoss = nn.MSELoss()
L1Loss = nn.L1Loss()
Loss_ssim = kornia.losses.SSIMLoss(11, reduction='mean')

trainloader = DataLoader(H5Dataset(r"data/train_hunandata2.h5"),
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

loader = {'train': trainloader, }
timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()

for epoch in range(num_epochs):
    for i, (data_Opt, data_SAR) in enumerate(loader['train']):
        data_Opt, data_SAR = data_Opt.cuda(), data_SAR.cuda()
        DIDF_Encoder_Opt.train()
        DIDF_Encoder_SAR.train()
        DIDF_Decoder.train()
        BaseFuseLayer.train()
        DetailFuseLayer_SAR.train()
        DetailFuseLayer_Opt.train()

        DIDF_Encoder_Opt.zero_grad()
        DIDF_Encoder_SAR.zero_grad()
        DIDF_Decoder.zero_grad()
        BaseFuseLayer.zero_grad()
        DetailFuseLayer_SAR.zero_grad()
        DetailFuseLayer_Opt.zero_grad()

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        optimizer3.zero_grad()
        optimizer4.zero_grad()
        optimizer5.zero_grad()
        optimizer6.zero_grad()

        if epoch < epoch_gap:
            feature_Opt_B, feature_Opt_D, _ = DIDF_Encoder_Opt(data_Opt)
            feature_SAR_B, feature_SAR_D, _ = DIDF_Encoder_SAR(data_SAR)
            data_Opt_hat, _ = DIDF_Decoder(data_Opt, feature_Opt_B, feature_Opt_D)
            data_SAR_hat, _ = DIDF_Decoder(data_SAR, feature_SAR_B, feature_SAR_D)

            cc_loss_B = cc(feature_Opt_B, feature_SAR_B)
            cc_loss_D = cc(feature_Opt_D, feature_SAR_D)
            mse_loss_Opt = 5 * Loss_ssim(data_Opt, data_Opt_hat) + MSELoss(data_Opt, data_Opt_hat)
            mse_loss_SAR = 5 * Loss_ssim(data_SAR, data_SAR_hat) + MSELoss(data_SAR, data_SAR_hat)

            Gradient_loss = L1Loss(kornia.filters.SpatialGradient()(data_Opt),
                                   kornia.filters.SpatialGradient()(data_Opt_hat))

            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)

            loss = coeff_mse_loss_VF * mse_loss_Opt + coeff_mse_loss_IF * mse_loss_SAR + coeff_decomp * loss_decomp + coeff_tv * Gradient_loss

            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_SAR.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_Opt.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer5.step()
        else:
            feature_Opt_B, feature_Opt_D, feature_Opt = DIDF_Encoder_Opt(data_Opt)
            feature_SAR_B, feature_SAR_D, feature_SAR = DIDF_Encoder_SAR(data_SAR)
            feature_F_B = BaseFuseLayer(feature_SAR_B + feature_Opt_B)
            feature_F_D1 = DetailFuseLayer_Opt(feature_Opt_D)
            feature_F_D2 = DetailFuseLayer_SAR(feature_SAR_D)
            feature_F_D = feature_F_D1 + feature_F_D2
            data_Fuse, feature_F = DIDF_Decoder(data_Opt, feature_F_B, feature_F_D)

            mse_loss_Opt = 5 * Loss_ssim(data_Opt, data_Fuse)
            mse_loss_SAR = 5 * Loss_ssim(data_SAR, data_Fuse)

            cc_loss_B = cc(feature_Opt_B, feature_SAR_B)
            cc_loss_D = cc(feature_Opt_D, feature_SAR_D)
            loss_decomp = (cc_loss_D) ** 2 / (1.01 + cc_loss_B)
            fusionloss, _, _ = criteria_fusion(data_Opt, data_SAR, data_Fuse)

            loss = fusionloss + coeff_decomp * loss_decomp
            loss.backward()
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_Opt.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Encoder_SAR.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DIDF_Decoder.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                BaseFuseLayer.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer_Opt.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            nn.utils.clip_grad_norm_(
                DetailFuseLayer_SAR.parameters(), max_norm=clip_grad_norm_value, norm_type=2)
            optimizer1.step()
            optimizer2.step()
            optimizer3.step()
            optimizer4.step()
            optimizer5.step()
            optimizer6.step()

        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] ETA: %.10s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                time_left,
            )
        )

    scheduler1.step()
    scheduler2.step()
    scheduler5.step()
    if not epoch < epoch_gap:
        scheduler3.step()
        scheduler4.step()
        scheduler6.step()

    if optimizer1.param_groups[0]['lr'] <= 1e-6:
        optimizer1.param_groups[0]['lr'] = 1e-6
    if optimizer2.param_groups[0]['lr'] <= 1e-6:
        optimizer2.param_groups[0]['lr'] = 1e-6
    if optimizer3.param_groups[0]['lr'] <= 1e-6:
        optimizer3.param_groups[0]['lr'] = 1e-6
    if optimizer4.param_groups[0]['lr'] <= 1e-6:
        optimizer4.param_groups[0]['lr'] = 1e-6
    if optimizer5.param_groups[0]['lr'] <= 1e-6:
        optimizer5.param_groups[0]['lr'] = 1e-6
    if optimizer6.param_groups[0]['lr'] <= 1e-6:
        optimizer6.param_groups[0]['lr'] = 1e-6

if True:
    checkpoint = {
        'DIDF_Encoder_Opt': DIDF_Encoder_Opt.state_dict(),
        'DIDF_Encoder_SAR': DIDF_Encoder_SAR.state_dict(),
        'DIDF_Decoder': DIDF_Decoder.state_dict(),
        'BaseFuseLayer': BaseFuseLayer.state_dict(),
        'DetailFuseLayer_SAR': DetailFuseLayer_SAR.state_dict(),
        'DetailFuseLayer_Opt': DetailFuseLayer_Opt.state_dict(),
    }
    torch.save(checkpoint, os.path.join("models/" + 'MDCANet' + '.pth'))