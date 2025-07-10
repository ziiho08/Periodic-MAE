import os
import numpy as np
import torch
import torch.optim as optim
import random
from tqdm import tqdm
from evaluation.post_process import calculate_hr
from evaluation.metrics import calculate_metrics
from neural_methods.model.PeriodicMAE import PeriodicMAE
from neural_methods.trainer.BaseTrainer import BaseTrainer
from neural_methods.loss.TorchLossComputer import PeriodicMAE_Loss
from torch.nn import MSELoss, L1Loss
from torch.cuda.amp import GradScaler
import torch.backends.cudnn as cudnn
import builtins
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from einops import rearrange
from neural_methods.loss.SiNCLoss import IPR_SSL, EMD_SSL,SNR_SSL, torch_power_spectral_density, add_noise_to_constants
import math
import matplotlib.pyplot as plt
scaler = GradScaler()

class PeriodicMAETrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.chunk_len = config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH
        self.warmup_epochs = 3
        self.min_lr = 1e-5
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0
        self.diff_flag = 0
        self.recon_loss = MSELoss()
        self.pos_loss = L1Loss()
        self.patch_size: int = 16
        self.normlize_target: bool = True,
        self.learning_rate = config.TRAIN.LR
        if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE == "DiffNormalized":
            self.diff_flag = 1
        if config.TOOLBOX_MODE == "train_and_test":
            self.model = PeriodicMAE().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
            
            if self.config.TRAIN.PRETRAIN:
                weight_decay = 0.05
                self.num_train_batches = len(data_loader["train"])
                scaled_lr = self.learning_rate * self.batch_size / 256
                self.min_lr = self.min_lr * self.batch_size / 256

                self.optimizer = optim.AdamW([
                    {'params': self.model.module.encoder.parameters(), 'lr': scaled_lr},    # Encoder with base learning rate
                    {'params': self.model.module.decoder.parameters(), 'lr': scaled_lr},    # Decoder with base learning rate
                ], betas=(0.9, 0.99), weight_decay=weight_decay)

                lr_lambdas = [self._cosine_scheduler_fn, self._cosine_scheduler_fn, self.lr_lambda_convblocklast]
                self.lr_scheduler_factors = self._cosine_scheduler_factors()
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambdas)
                print(f"Initial Learning Rate: {scaled_lr}")

            else: 
                self.num_train_batches = len(data_loader["train"])
                pretrain_model_path = config.TRAIN.PRETRAIN_MODEL
                print("Using pretrained model!")
                print(pretrain_model_path)

                checkpoint = torch.load(pretrain_model_path)
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = self.model.state_dict()
                
                if next(iter(model_dict)).startswith('module.') and not next(iter(pretrained_dict)).startswith('module.'):
                    pretrained_dict = {'module.' + k: v for k, v in pretrained_dict.items()}
                elif not next(iter(model_dict)).startswith('module.') and next(iter(pretrained_dict)).startswith('module.'):
                    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}

                pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
                print(f"Length of Pretrained parameters that will be loaded: {len(pretrained_dict_filtered.keys())}")

                model_dict.update(pretrained_dict_filtered)
                self.model.load_state_dict(model_dict, strict=True)

                for name, param in self.model.named_parameters():
                    if 'decoder' in name:
                        param.requires_grad = False

                self.criterion = PeriodicMAE_Loss() 
                weight_decay = 0.0
                self.optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=config.TRAIN.LR, betas=(0.9, 0.95), weight_decay=weight_decay) #betas=(0.9, 0.99)
                self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)

        elif config.TOOLBOX_MODE == "only_test":
            self.model = PeriodicMAE().to(self.device)
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        else:
            raise ValueError("EfficientPhys trainer initialized in incorrect toolbox mode!")
        
    def lr_lambda_convblocklast(self, epoch):
        return max(0.95 ** epoch, 1e-5)
    
    def _cosine_scheduler_factors(self):
        warmup_schedule = np.array([])
        warmup_iters = self.warmup_epochs * self.num_train_batches
        if self.warmup_epochs > 0:
            warmup_schedule = np.linspace(0, self.learning_rate, warmup_iters)

        iters = np.arange(self.max_epoch_num  * self.num_train_batches - warmup_iters)
        schedule = np.array(
            [self.min_lr + 0.5 * (self.learning_rate - self.min_lr) * (1 + math.cos(math.pi * i / (len(iters))))
                for i in iters])

        schedule = np.concatenate((warmup_schedule, schedule))

        assert len(schedule) == self.max_epoch_num  * self.num_train_batches
        values_factors = schedule / self.learning_rate
        return values_factors

    def _cosine_scheduler_fn(self, epoch):
        return self.lr_scheduler_factors[epoch]

    def train(self, data_loader):

        if self.config.TRAIN.RESUME_PRETRAIN:
            checkpoint_path = ""
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            start_epoch = checkpoint['epoch'] + 1
        else:
            start_epoch = 0

        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")
        
        mean_valid_losses = []
        for epoch in range(start_epoch, self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=150)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                data, labels, speed, masks = batch[0].float(), batch[1].float(), batch[4].float(), batch[5].bool()
                N, D, C, H, W = data.shape
                self.model = self.model.to(self.device)
                masks = masks.to(self.device).flatten(1).to(torch.bool)
                
                if self.config.TRAIN.PRETRAIN:
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    with torch.no_grad():
                        image = data.permute(0,2,1,3,4)
                        mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :,None,None,None]
                        std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :,None, None,None]
                        unnorm_images = image * std + mean  # in [0, 1]
            
                        if self.normlize_target:
                            images_squeeze = rearrange(
                                unnorm_images,
                                'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c',
                                p0=1,
                                p1=self.patch_size,
                                p2=self.patch_size)
                            images_norm = (images_squeeze - images_squeeze.mean(
                                dim=-2, keepdim=True)) / (
                                    images_squeeze.var(
                                        dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                            images_patch = rearrange(images_norm, 'b n p c -> b n (p c)')
                        else:
                            images_patch = rearrange(
                                unnorm_images,
                                'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)',
                                p0=1,
                                p1=self.patch_size,
                                p2=self.patch_size)
                    B_, N_, C_ = images_patch.shape
                    label_patch = images_patch[~masks].reshape(B_, -1, C_)
                    
                # Forward pass
                self.optimizer.zero_grad()
                losses_dict = {}
                total_loss = 0.0
                if self.config.TRAIN.PRETRAIN:
                    pred_patch, pred_ppg = self.model(data, masks)
                    rec_loss = (pred_patch - label_patch) ** 2
                    rec_loss = rec_loss.mean(dim=-1)

                    cal_loss_mask = ~masks[~masks].reshape(B_, -1)
                    rec_loss = (rec_loss * cal_loss_mask).sum() / cal_loss_mask.sum()
                    total_loss += rec_loss * 1.0
                    losses_dict['r'] = rec_loss.item()

                    fps = float(30)
                    low_hz = float(0.6666667)
                    high_hz = float(3.0)
                    pred_ppg = add_noise_to_constants(pred_ppg)
                    freqs, psd = torch_power_spectral_density(pred_ppg, fps=fps, low_hz=low_hz, high_hz=high_hz, normalize=False, bandpass=False)
                    
                    freqs = freqs.to(self.device)
                    psd = psd.to(self.device)

                    # Bandwidth loss (IPR_SSL)
                    bandwidth_loss = IPR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=self.device)
                    total_loss += bandwidth_loss * 1.0
                    losses_dict['b'] = bandwidth_loss.item()
                    
                    # Sparsity loss (SNR_SSL)
                    sparsity_loss = SNR_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=self.device)
                    total_loss += sparsity_loss * 1.0
                    losses_dict['s'] = sparsity_loss.item()
                    
                    # Variance loss (EMD_SSL)
                    variance_loss = EMD_SSL(freqs, psd, speed=speed, low_hz=low_hz, high_hz=high_hz, device=self.device)
                    total_loss += variance_loss * 1.0
                    losses_dict['v'] = variance_loss.item()

                    total_loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    tbar.set_postfix(loss=total_loss.item(), **losses_dict)

                else: 
                    if self.config.TRAIN.AUG :
                        data,labels = self.data_augmentation(data,labels)
                    
                    data = data.to(self.device)
                    labels = labels.to(self.device)
                    
                    self.optimizer.zero_grad()
                    pred_ppg, _ = self.model(data, masks)
                    pred_ppg = (pred_ppg - torch.mean(pred_ppg, axis=-1).view(-1, 1)) / torch.std(pred_ppg, axis=-1).view(-1, 1)
                    loss = 0.0
                    for ib in range(N):
                        loss += self.criterion(pred_ppg[ib], labels[ib], epoch, self.config.TRAIN.DATA.FS, self.diff_flag, self.device)
                    loss = loss / N
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    tbar.set_postfix(loss=loss.item())
            
            self.save_model(epoch)
            if self.config.TRAIN.PRETRAIN == False:
                if not self.config.TEST.USE_LAST_EPOCH:
                    valid_loss = self.valid(data_loader)
                    mean_valid_losses.append(valid_loss)
                    print('validation loss: ', valid_loss)
                    if self.min_valid_loss is None or valid_loss < self.min_valid_loss:
                        self.min_valid_loss = valid_loss
                        self.best_epoch = epoch
                        print(f"Update best model! Best epoch: {self.best_epoch}")
        
        if self.config.TRAIN.PRETRAIN == False:
            if not self.config.TEST.USE_LAST_EPOCH:
                print(f"best trained epoch: {self.best_epoch}, min_val_loss: {self.min_valid_loss}")  

    def test(self, data_loader):
        if self.config.TRAIN.PRETRAIN == False:
            """ Model evaluation on the testing dataset."""
            if data_loader["test"] is None:
                raise ValueError("No data for test")

            print('')
            print("===Testing===")
            if self.config.TOOLBOX_MODE == "only_test":
                if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                    raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
                checkpoint = torch.load(self.config.INFERENCE.MODEL_PATH)
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = self.model.state_dict()
                model_dict.update(pretrained_dict)
                self.model.load_state_dict(model_dict, strict=False)
                print("Testing uses pretrained model!")
                print(self.config.INFERENCE.MODEL_PATH)
            else:
                if self.config.TEST.USE_LAST_EPOCH:
                    last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                    print("Testing uses last epoch as non-pretrained model!")
                    print(last_epoch_model_path)
                    checkpoint = torch.load(last_epoch_model_path)
                    pretrained_dict = checkpoint['model_state_dict']
                    model_dict = self.model.state_dict()
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict, strict=False)

                else:
                    best_model_path = os.path.join(
                        self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                    print("Testing uses best epoch selected using model selection as non-pretrained model!")
                    print(best_model_path)

                    checkpoint = torch.load(best_model_path)
                    pretrained_dict = checkpoint['model_state_dict']
                    model_dict = self.model.state_dict()
                    model_dict.update(pretrained_dict)
                    self.model.load_state_dict(model_dict, strict=False)

            self.model = self.model.to(self.config.DEVICE)
            self.model.eval()
            with torch.no_grad():
                predictions = dict()
                labels = dict()
                for _, test_batch in enumerate(data_loader['test']):
                    batch_size = test_batch[0].shape[0]
                    chunk_len = self.chunk_len
                    data_test, labels_test, masks_test = test_batch[0].to(self.config.DEVICE), test_batch[1].to(self.config.DEVICE), test_batch[-1].to(self.config.DEVICE)
                    pred_ppg_test, _ = self.model(data_test, masks_test)
                    pred_ppg_test = (pred_ppg_test-torch.mean(pred_ppg_test, axis=-1).view(-1, 1))/torch.std(pred_ppg_test, axis=-1).view(-1, 1)    # normalize
                    labels_test = labels_test.view(-1, 1)
                    pred_ppg_test = pred_ppg_test.view( -1 , 1)
                    for ib in range(batch_size):
                        subj_index = test_batch[2][ib]
                        sort_index = int(test_batch[3][ib])
                        if subj_index not in predictions.keys():
                            predictions[subj_index] = dict()
                            labels[subj_index] = dict()
                        predictions[subj_index][sort_index] = pred_ppg_test[ib * chunk_len:(ib + 1) * chunk_len]
                        labels[subj_index][sort_index] = labels_test[ib * chunk_len:(ib + 1) * chunk_len]
                print(' ')
                calculate_metrics(predictions, labels, self.config)
                if self.config.TEST.OUTPUT_SAVE_DIR: # saving test outputs
                    self.save_test_outputs(predictions, labels, self.config)
        print("=======TRAIN FININSHED=======")

    def valid(self, data_loader):
        """ Model evaluation on the validation dataset."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")
        print("===Validating===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                data_valid, labels_valid, masks_valid = valid_batch[0].to(self.device), valid_batch[1].to(self.device), valid_batch[-1].to(self.config.DEVICE)
                pred_ppg_valid = self.model(data_valid, masks_valid)
                pred_ppg_valid = (pred_ppg_valid - torch.mean(pred_ppg_valid, axis=-1).view(-1, 1)) / torch.std(pred_ppg_valid, axis=-1).view(-1, 1)  # normalize
                for ib in range(data_valid.size(0)):
                    loss = self.criterion(pred_ppg_valid[ib], labels_valid[ib], self.config.TRAIN.EPOCHS, self.config.VALID.DATA.FS, self.diff_flag, self.device)
                    valid_loss.append(loss.item())
        return np.mean(valid_loss)

    def save_model(self, epoch):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(self.model_dir, f"{self.model_file_name}_Epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, model_path)
        print(f'Saved Model Path: {model_path}')
    
    def data_augmentation(self,data,labels):
        N, D, C, H, W = data.shape
        data_aug = np.zeros((N, D, C, H, W))
        labels_aug = np.zeros((N, D))
        for idx in range(N):
            gt_hr_fft, _  = calculate_hr(labels[idx], labels[idx] , diff_flag = self.diff_flag , fs=self.config.VALID.DATA.FS)
            rand1 = random.random()
            rand2 = random.random()
            rand3 = random.randint(0, D//2-1)
            if rand1 < 0.5 :
                if gt_hr_fft > 90 :
                    for tt in range(rand3,rand3+D):
                        if tt%2 == 0:
                            data_aug[idx,tt-rand3,:,:,:] = data[idx,tt//2,:,:,:]
                            labels_aug[idx,tt-rand3] = labels[idx,tt//2]                    
                        else:
                            data_aug[idx,tt-rand3,:,:,:] = data[idx,tt//2,:,:,:]/2 + data[idx,tt//2+1,:,:,:]/2
                            labels_aug[idx,tt-rand3] = labels[idx,tt//2]/2 + labels[idx,tt//2+1]/2
                elif gt_hr_fft < 75 :
                    for tt in range(D):
                        if tt < D/2 :
                            data_aug[idx,tt,:,:,:] = data[idx,tt*2,:,:,:]
                            labels_aug[idx,tt] = labels[idx,tt*2] 
                        else :                                    
                            data_aug[idx,tt,:,:,:] = data_aug[idx,tt-D//2,:,:,:]
                            labels_aug[idx,tt] = labels_aug[idx,tt-D//2] 
                else :
                    data_aug[idx] = data[idx]
                    labels_aug[idx] = labels[idx]                                          
            else :
                data_aug[idx] = data[idx]
                labels_aug[idx] = labels[idx]
        data_aug = torch.tensor(data_aug).float()
        labels_aug = torch.tensor(labels_aug).float()
        if rand2 < 0.5:
            data_aug = torch.flip(data_aug, dims=[4])
        data = data_aug
        labels = labels_aug
        return data,labels