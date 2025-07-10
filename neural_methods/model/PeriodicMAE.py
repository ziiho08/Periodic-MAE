'''
  Modifed based on the RhythmFormer here: https://github.com/zizheng-guo/RhythmFormer
'''
from typing import Optional
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from typing import Tuple, Union
from timm.models.layers import trunc_normal_, to_2tuple
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from neural_methods.model.base.video_bra import video_BiFormerBlock
from neural_methods.model.base.mae_utils import Block
from config import get_config
import numpy as np
from .base.mae_utils import *
from main import *
parser = argparse.ArgumentParser()
parser = add_args(parser)
parser = trainer.BaseTrainer.BaseTrainer.add_trainer_args(parser)
parser = data_loader.BaseLoader.BaseLoader.add_data_loader_args(parser)
args = parser.parse_args()

# configurations.
config = get_config(args)
print('Configuration:')
print(config, end='\n\n')

class Fusion_Stem(nn.Module):
    def __init__(self,apha=0.5,belta=0.5):
        super(Fusion_Stem, self).__init__()

        self.stem11 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )
        
        self.stem12 = nn.Sequential(nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            )

        self.stem21 =nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stem22 =nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.apha = apha
        self.belta = belta

    def forward(self, x):
        """Definition of Fusion_Stem.
        Args:
          x [N,D,C,H,W]
        Returns:
          fusion_x [N*D,C,H/4,W/4]
        """
        N, D, C, H, W = x.shape
        x1 = torch.cat([x[:,:1,:,:,:],x[:,:1,:,:,:],x[:,:D-2,:,:,:]],1)
        x2 = torch.cat([x[:,:1,:,:,:],x[:,:D-1,:,:,:]],1)
        x3 = x
        x4 = torch.cat([x[:,1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x5 = torch.cat([x[:,2:,:,:,:],x[:,D-1:,:,:,:],x[:,D-1:,:,:,:]],1)
        x_diff = self.stem12(torch.cat([x2-x1,x3-x2,x4-x3,x5-x4],2).view(N * D, 12, H, W))
        x3 = x3.contiguous().view(N * D, C, H, W)
        x = self.stem11(x3)

        #fusion layer1
        x_path1 = self.apha*x + self.belta*x_diff
        x_path1 = self.stem21(x_path1)
        #fusion layer2
        x_path2 = self.stem22(x_diff)
        x = self.apha*x_path1 + self.belta*x_path2
        
        return x
    
class TPT_Block(nn.Module):
    def __init__(self, dim, depth, num_heads, t_patch, topk,
                 mlp_ratio=4., drop_path=0., side_dwconv=5):
        super().__init__()
        self.dim = dim
        self.depth = depth
        ############ downsample layers & upsample layers #####################
        self.downsample_layers = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.layer_n = int(math.log(t_patch,2))
        for i in range(self.layer_n):
            downsample_layer = nn.Sequential(
                nn.BatchNorm3d(dim), 
                nn.Conv3d(dim , dim , kernel_size=(2, 1, 1), stride=(2, 1, 1)),
                )
            self.downsample_layers.append(downsample_layer)
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=(2, 1, 1)),
                nn.Conv3d(dim , dim , [3, 1, 1], stride=1, padding=(1, 0, 0)),   
                nn.BatchNorm3d(dim),
                nn.ELU(),
                )
            self.upsample_layers.append(upsample_layer)
        ######################################################################
        self.blocks = nn.ModuleList([
            video_BiFormerBlock(
                    dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    num_heads=num_heads,
                    t_patch=t_patch,
                    topk=topk,
                    mlp_ratio=mlp_ratio,
                    side_dwconv=side_dwconv,
                )
            for i in range(depth)
        ])
    def forward(self, x:torch.Tensor):
        """Definition of TPT_Block.
        Args:
          x [N,C,D,H,W]
        Returns:
          x [N,C,D,H,W]
        """
        attention_weights = []
        for i in range(self.layer_n) :
            x = self.downsample_layers[i](x)
        for blk in self.blocks:
            x, attn_weights = blk(x)
            attention_weights.append(attn_weights)
        for i in range(self.layer_n) :
            x = self.upsample_layers[i](x)

        return x, attention_weights  
    
    
class Encoder(nn.Module):
    def __init__(
        self, 
        dim: int = 64, frame: int = 160,
        image_size: Optional[int] = (160,128,128),
        in_chans=64, head_dim=16,
        stage_n = 3,
        embed_dim=[64, 64, 64], mlp_ratios=[1.5, 1.5, 1.5],
        depth=[2, 2, 2], 
        t_patchs:Union[int, Tuple[int]]=(2, 4, 8),
        topks:Union[int, Tuple[int]]=(40, 40, 40),
        side_dwconv:int=3,
        drop_path_rate=0.,
    ):
        super().__init__()
        self.pretrain = config.TRAIN.PRETRAIN,
        self.image_size = image_size  
        self.frame = frame  
        self.dim = dim              
        self.stage_n = stage_n

        self.Fusion_Stem = Fusion_Stem()
        self.patch_embedding = nn.Conv3d(in_chans, embed_dim[0], kernel_size=(1, 4, 4), stride=(1, 4, 4))
        self.ConvBlockLast = nn.Conv1d(embed_dim[-1], 1, kernel_size=1,stride=1, padding=0)
        self.fc_layer_1 = nn.Linear(embed_dim[-1], 64)
        self.fc_layer_2 = nn.Linear(64, 32)
        self.fc_layer_3 = nn.Linear(32, 1)
        ##########################################################################
        nheads= [dim // head_dim for dim in embed_dim]
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]

        self.stages = nn.ModuleList()
        for i in range(self.stage_n):
            stage = TPT_Block(dim=embed_dim[i],
                               depth=depth[i],
                               num_heads=nheads[i], 
                               mlp_ratio=mlp_ratios[i],
                               drop_path=dp_rates[sum(depth[:i]):sum(depth[:i+1])],
                               t_patch=t_patchs[i], topk=topks[i], side_dwconv=side_dwconv
                               )
            self.stages.append(stage)
        ##########################################################################
        self.num_features = self.embed_dim = embed_dim #MAE embed            
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def rppg_head(self, x):
        features_last = torch.mean(x,3)    #[N, 64, D, 8]  
        features_last = torch.mean(features_last,3)    #[N, 64, D]  

        features_last = features_last.permute(0, 2, 1)  # [N, D, 64]
        features_last = self.fc_layer_1(features_last)    # [N, D, 1]        
        features_last = self.fc_layer_2(features_last)    # [N, D, 1]  
        rPPG = self.fc_layer_3(features_last)    # [N, D, 1]  
        rPPG = rPPG.squeeze(-1)                 # [N, D]
        return rPPG
    
    def forward_features(self, x, mask):
        N, D, C, H, W = x.shape # ([B, 160, 3, 128, 128])
        x = self.Fusion_Stem(x)    #[N*D, 64, H/4, W/4]
        x = x.view(N,D,64,H//4,W//4).permute(0,2,1,3,4) #[B, 64, 160, 32, 32]
        x = self.patch_embedding(x)   #[B, 64, 160, 8, 8]

        N, C, D, H, W = x.shape
        if self.pretrain[0] == True:
            x = x.flatten(2).transpose(1,2)
            N,_,C = x.shape
            x = x[mask].reshape(N,C,-1,H,W)
        
        for i in range(self.stage_n):
            x = self.stages[i](x)    
        return x
    
    def forward(self, x, mask):
        x = self.forward_features(x, mask)
        x = self.rppg_head(x)
        return x
    
class Decoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
        Source code: https://github.com/MCG-NJU/VideoMAE
    """
    def __init__(self,
                 patch_size=16,
                 num_classes=None,
                 embed_dim=64,
                 depth=8,
                 num_heads=4,
                 mlp_ratio=3.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 num_patches=196,
                 tubelet_size=1,
                 cos_attn=False):
        super().__init__()

        decoder_embed_dim = 3 * tubelet_size * patch_size * patch_size
        self.num_features = self.embed_dim = embed_dim
        self.patch_size = patch_size

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                cos_attn=cos_attn) for i in range(depth)
        ])        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(
            embed_dim, decoder_embed_dim) if decoder_embed_dim > 0 else nn.Identity()
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def get_classifier(self):
        return self.head

    def reset_classifier(self, decoder_embed_dim, global_pool=''):
        self.decoder_embed_dim = decoder_embed_dim
        self.head = nn.Linear(
            self.embed_dim, decoder_embed_dim) if decoder_embed_dim > 0 else nn.Identity()
        
    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:]))
        else:
            x = self.head(self.norm(x))
        return x
    
class PeriodicMAE(nn.Module):

    def __init__(
        self,
        img_size=128,
        patch_size=16,
        encoder_in_chans=64,
        encoder_num_classes=0,
        encoder_embed_dim=64,
        decoder_num_classes=768, 
        decoder_embed_dim=768,
        decoder_depth=4,
        decoder_num_heads=3,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_values=0.,
        use_learnable_pos_emb=False,
        tubelet_size=1,
        all_frames=160,
        cos_attn=False,
    ):
        super().__init__()
        self.pretrain = config.TRAIN.PRETRAIN
        self.encoder = Encoder(
            in_chans= encoder_in_chans,
            dim = 64, 
            frame = 160,
            image_size = (160,128,128),
            head_dim=16,
            stage_n = 3,
            embed_dim=[64, 64, 64],  
            mlp_ratios=[1.5, 1.5, 1.5], 
            depth=[2, 2, 2],  
            t_patchs =(2, 4, 8), 
            topks =(40, 40, 40),
            side_dwconv = 3,
            drop_path_rate=0.0,
            use_checkpoint_stages=[],
)

        self.decoder = Decoder(
            patch_size=patch_size,
            num_classes=decoder_num_classes,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size,
            cos_attn=cos_attn)
    
        
        self.encoder_num_patches = all_frames*(img_size//patch_size)*(img_size//patch_size)
        self.encoder_to_decoder = nn.Linear(
            encoder_embed_dim, decoder_embed_dim, bias=False)

        self.decoder_to_rppg = nn.Linear(
            decoder_embed_dim, encoder_embed_dim, bias=False)
        
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.pos_embed = get_sinusoid_encoding_table(
            self.encoder_num_patches, decoder_embed_dim)
        
        self.ConvBlockLast = nn.Conv1d(encoder_embed_dim, 1, kernel_size=1,stride=1, padding=0)
        no_grad_trunc_normal_(self.mask_token, mean=0., std=0.02, a=-0.02, b=0.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    def rppg_head(self, x):
        features_last = torch.mean(x,3)   
        features_last = torch.mean(features_last,3)
        rPPG = self.ConvBlockLast(features_last)
        return rPPG

    def forward(self, x, mask, decode_mask=None):
        if self.pretrain == True:
            decode_vis = ~mask if decode_mask is None else decode_mask
            x_vis_enc = self.encoder.forward_features(x, mask)

            B, enc_C, enc_T, enc_H, enc_W = x_vis_enc.shape
            x_vis_enc = x_vis_enc.view(B,enc_C,-1).permute(0,2,1)
            x_vis = self.encoder_to_decoder(x_vis_enc)
            B, _, C = x_vis.shape

            expand_pos_embed = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
            pos_emd_vis = expand_pos_embed[mask].reshape(B, -1, C)
            pos_emd_mask = expand_pos_embed[decode_vis].reshape(B, -1, C)

            x_full = torch.cat(
                [x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)

            x = self.decoder(x_full, pos_emd_mask.shape[1])

            x_for_rppg = torch.cat(
                        [x_vis + pos_emd_vis, x + pos_emd_mask], dim=1)
            
            x_rppg = self.decoder_to_rppg(x_for_rppg)
            x_rppg = x_rppg.permute(0, 2, 1).reshape(B, enc_C, x_rppg.shape[1] // 64, enc_H, enc_W)
            rppg = self.rppg_head(x_rppg)
            return x, rppg
        
        else:
            rppg = self.encoder(x, mask)
            return rppg


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=128,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=512,
                 num_frames=160,
                 tubelet_size=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_spatial_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        num_patches = num_spatial_patches * (num_frames // tubelet_size)

        self.img_size = img_size
        self.tubelet_size = tubelet_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
            stride=(self.tubelet_size, patch_size[0], patch_size[1]))

    def forward(self, x, **kwargs):
        B, C, T, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[
            1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # b, c, l -> b, l, c
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

def no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor