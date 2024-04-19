# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# iRPS: https://github.com/microsoft/Cream/tree/main/iRPE
# --------------------------------------------------------

from functools import partial
from re import T
from typing import MutableMapping
from unittest.mock import patch

# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from vision_transformer_irpe import PatchEmbed, Block
import math
from util.pos_embed import get_2d_sincos_pos_embed
from typing import Tuple, Union

def get_gkern(kernlen, std):
    """Returns a 2D Gaussian kernel array."""

    def _gaussian_fn(kernlen, std):
        n = torch.arange(0, kernlen).float()
        n -= n.mean()
        n /= std
        w = torch.exp(-0.5 * n**2)
        return w

    gkern1d = _gaussian_fn(kernlen, std)
    gkern2d = torch.outer(gkern1d, gkern1d)
    return gkern2d / gkern2d.sum()

class HOGLayerC(nn.Module):
    def __init__(self, nbins=9, pool=7, gaussian_window=0):
        super(HOGLayerC, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        weight_x = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        weight_x = weight_x.view(1, 1, 3, 3)
        weight_y = weight_x.transpose(2, 3)
        self.register_buffer("weight_x", weight_x)
        self.register_buffer("weight_y", weight_y)

        self.gaussian_window = gaussian_window
        if gaussian_window:
            gkern = get_gkern(gaussian_window, gaussian_window // 2)
            self.register_buffer("gkern", gkern)

    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(1, 1, 1, 1), mode="reflect")
        gx_rgb = F.conv2d(
            x, self.weight_x, bias=None, stride=1, padding=0, groups=1
        )
        gy_rgb = F.conv2d(
            x, self.weight_y, bias=None, stride=1, padding=0, groups=1
        )
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)
        # phase = torch.atan2(gx_rgb, gy_rgb)
        # phase = phase / self.pi * self.nbins  # [-9, 9]


        return norm_rgb  # B 1 nbins H W

class GF(nn.Module):
    def __init__(self, nbins=9, pool=7, kensize=5, img_size=224, patch_size=16):
        super(GF, self).__init__()
        self.nbins = nbins
        self.pool = pool
        self.pi = math.pi
        self.img_size = img_size
        self.patch_size = patch_size
        self.k = kensize

        # def creat_gauss_kernel(r=1, sigma=-1):
        #     if sigma <= 0:
        #         sigma = 0.3 * ((2*r+1 - 1) * 0.5 - 1) + 0.8
        #
        #     X = np.linspace(-r, r, 2*r+1)
        #     Y = np.linspace(-r, r, 2*r+1)
        #     x, y = np.meshgrid(X, Y)
        #     x0 = 0
        #     y0 = 0
        #     gauss = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))
        #
        #     M_13 = np.concatenate([np.ones([r, 2*r+1]), np.zeros([r+1, 2*r+1])], axis=0)
        #     M_23 = np.concatenate([np.zeros([r+1, 2 * r + 1]), np.ones([r, 2 * r + 1])], axis=0)
        #
        #     M_11 = np.concatenate([np.ones([2*r+1, r]), np.zeros([2*r+1, r+1])], axis=1)
        #     M_21 = np.concatenate([np.zeros([2 * r + 1, r+1]), np.ones([2 * r + 1, r])], axis=1)
        #
        #     return torch.from_numpy((gauss*M_13)).float(), torch.from_numpy((gauss*M_23)).float(), torch.from_numpy((gauss*M_11)).float(), torch.from_numpy((gauss*M_21)).float()
        #
        def creat_kernel(r=1):

            M_13 = np.concatenate([np.ones([r+1, 2*r+1]), np.zeros([r, 2*r+1])], axis=0)
            M_23 = np.concatenate([np.zeros([r, 2 * r + 1]), np.ones([r+1, 2 * r + 1])], axis=0)

            M_11 = np.concatenate([np.ones([2*r+1, r+1]), np.zeros([2*r+1, r])], axis=1)
            M_21 = np.concatenate([np.zeros([2 * r + 1, r]), np.ones([2 * r + 1, r+1])], axis=1)


            return torch.from_numpy((M_13)).float(), torch.from_numpy((M_23)).float(), torch.from_numpy((M_11)).float(), torch.from_numpy((M_21)).float()

        M13, M23, M11, M21 = creat_kernel(self.k)

        weight_x1 = M11.view(1, 1, self.k*2+1, self.k*2+1)
        weight_x2 = M21.view(1, 1, self.k*2+1, self.k*2+1)

        weight_y1 = M13.view(1, 1, self.k*2+1, self.k*2+1)
        weight_y2 = M23.view(1, 1, self.k*2+1, self.k*2+1)

        self.register_buffer("weight_x1", weight_x1)
        self.register_buffer("weight_x2", weight_x2)
        self.register_buffer("weight_y1", weight_y1)
        self.register_buffer("weight_y2", weight_y2)


    @torch.no_grad()
    def forward(self, x):
        # input is RGB image with shape [B 3 H W]
        x = F.pad(x, pad=(self.k, self.k, self.k, self.k), mode="reflect") + 1e-2
        gx_1 = F.conv2d(
            x, self.weight_x1, bias=None, stride=1, padding=0, groups=1
        )
        gx_2 = F.conv2d(
            x, self.weight_x2, bias=None, stride=1, padding=0, groups=1
        )
        gy_1 = F.conv2d(
            x, self.weight_y1, bias=None, stride=1, padding=0, groups=1
        )
        gy_2 = F.conv2d(
            x, self.weight_y2, bias=None, stride=1, padding=0, groups=1
        )
        gx_rgb = torch.log((gx_1) / (gx_2))
        gy_rgb = torch.log((gy_1) / (gy_2))
        norm_rgb = torch.stack([gx_rgb, gy_rgb], dim=-1).norm(dim=-1)

        # phase = torch.atan2(gx_rgb, gy_rgb)
        # phase = phase / self.pi * self.nbins  # [-9, 9]
        #
        # b, c, h, w = norm_rgb.shape
        # out = torch.zeros(
        #     (b, c, self.nbins, h, w), dtype=torch.float, device=x.device
        # )
        # phase = phase.view(b, c, 1, h, w)
        # norm_rgb = norm_rgb.view(b, c, 1, h, w)

        # plt.subplot(111)
        # plt.imshow(x[0].cpu().squeeze())
        # plt.axis('off')
        # plt.savefig("./origin.png", dpi=600, bbox_inches='tight',  pad_inches = 0.0)
        # plt.subplot(111)
        # plt.imshow(norm_rgb[0].cpu().squeeze())
        # plt.axis('off')
        # plt.savefig("./1.png", dpi=600, bbox_inches='tight',  pad_inches = 0.0)
        # plt.show()

        # out.scatter_add_(2, phase.floor().long() % self.nbins, norm_rgb)
        # # b, c, 9, h, w
        #
        # out = out.unfold(3, self.pool, self.pool)
        #
        # out = out.unfold(4, self.pool, self.pool)
        # # b, c, 9, 28, 28, self.pool, self.pool
        # out = out.sum(dim=[-1, -2])
        # # b, c, 9, 28, 28
        # out = torch.nn.functional.normalize(out, p=2, dim=2) # B 1 nbins H W
        # # b, c, 9, 28, 28
        # tmp_hog = out.flatten(1, 2)  # return B C H W
        # # b, 9, 28, 28
        # unfold_size = tmp_hog.shape[-1] // (self.img_size // self.patch_size)
        # # b, 9, 14, 14, 9, 2, 2
        # target = (
        #     tmp_hog.permute(0, 2, 3, 1)
        #         .unfold(1, unfold_size, unfold_size)
        #         .unfold(2, unfold_size, unfold_size)
        #         .flatten(1, 2)
        #         .flatten(2)
        # )

        return norm_rgb


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.encoder_pred = nn.Linear(embed_dim, decoder_embed_dim, bias=True) # decoder to patch
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, 256*4, bias=True)  # decoder to patch

        # --------------------------------------------------------------------------
        self.nbins = 9
        self.cell_sz = 8
        # self.sarfeature = HOGLayerC(nbins=self.nbins,
        #                           pool=self.cell_sz, )
        self.sarfeature1 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=5,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature2 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=9,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature3 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=13,
                                  img_size=self.img_size,patch_size=self.patch_size)
        self.sarfeature4 = GF(nbins=self.nbins,pool=self.cell_sz,kensize=17,
                                  img_size=self.img_size,patch_size=self.patch_size)

        # --------------------------------------------------------------------------
        # MAE decoder specifics

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def _get_pixel_label_2d(self, input_img, output_masks, norm=True):
        input_img = input_img.permute(0, 2, 3, 1)
        labels = []
        for depth, output_mask in zip(self.pretrain_depth, output_masks):
            size = self.feat_stride[depth][-1]
            label = input_img.unfold(1, size, size).unfold(2, size, size)
            label = label.flatten(1, 2).flatten(2)
            label = label[output_mask]
            if norm:
                mean = label.mean(dim=-1, keepdim=True)
                var = label.var(dim=-1, keepdim=True)
                label = (label - mean) / (var + 1.0e-6) ** 0.5
            labels.append(label)
        return labels

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def sample_patch_index_single_window(self,x,patch_index, keep_ratio):
        N, H, W, D = x.shape
        x = x.view(N,H*W,D)


        noise = torch.rand(N,patch_index.shape[0], device=patch_index.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise,dim=1)  # ascend: small is keep, large is remove

        ids_keep = ids_shuffle[:,:keep_ratio]

        patch_keeps = patch_index[ids_keep]

        return patch_keeps

    def sample_patch_index(self,x,patch_index, keep_ratio):

        N, H, W, D = x.shape
        M,P = patch_index.shape
        patch_index = patch_index.unsqueeze(0).expand(N,M,P)


        noise = torch.rand(N,M,P, device=patch_index.device)  # noise in [0, 1]
        ids_shuffle = torch.argsort(noise,dim=-1)  # ascend: small is keep, large is remove


        ids_keep = ids_shuffle[:,:,:keep_ratio]

        patch_keeps = torch.gather(patch_index, -1, ids_keep)

        return patch_keeps

    def generate_window_patches(self,x,left,top, window_size, mask_ratio):
        N, H, W, D = x.shape
        window_number = left.shape[0]
        

        #  extract the windows based on the coordinates
        left = left.unsqueeze(-1).expand(window_number,window_size)
        top  = top.unsqueeze(-1).expand(window_number, window_size)


        row = torch.arange(0,window_size,device=x.device).unsqueeze(0).expand(window_number,window_size)+left
        column = torch.arange(0,window_size*W,W, device = x.device).unsqueeze(0).expand(window_number, window_size)+top*W
        

        in_window_mask_number = int(window_size*window_size*mask_ratio)  

        assert in_window_mask_number>=1
        in_window_patches =row.unsqueeze(1).expand(window_number,window_size,window_size)  + column.unsqueeze(-1).expand(left.shape[0],window_size,window_size)
        in_window_patches = in_window_patches.view(window_number,-1)


        # sample the masked patch ids
        ids_mask_in_window =self.sample_patch_index(x,in_window_patches,in_window_mask_number)


        patches_to_keep = in_window_patches.unsqueeze(0).expand(N, window_number,window_size* window_size)
        x = x.view(N,H*W,D).unsqueeze(0).repeat(window_number,1, 1,1).view(N*window_number,H*W,D)


        sorted_patch_to_keep,_ = torch.sort(patches_to_keep,dim=-1)
        sorted_patch_to_keep = sorted_patch_to_keep.view(N*window_number,-1)

        ids_mask_in_window = ids_mask_in_window.view(N*window_number, -1)

        # gather the masked patches
        x_masked = torch.gather(x, dim=1, index=sorted_patch_to_keep.unsqueeze(-1).repeat(1, 1, D)).clone()
        # indices for recontruction
        mask_indices = ((sorted_patch_to_keep.unsqueeze(-1)- ids_mask_in_window.unsqueeze(1))==0).sum(-1)==1


        # zero out the patches in mask
        x_masked[mask_indices]=self.mask_token

 
        return x_masked, sorted_patch_to_keep,mask_indices


    def forward_encoder(self, x, window_size, num_window, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        x = x.type(torch.float32)

        N, _, C = x.shape
        H = W = self.img_size // self.patch_size
        x= x.view(N,H,W,C)
    

        assert window_size<= H and window_size <=W

        # sample window coordinates
        rand_top_locations = torch.randperm(H-window_size+1,device=x.device)[:num_window]
        rand_left_locations = torch.randperm(W-window_size+1,device=x.device)[:num_window]

        # generate the sampled and mask patches from the small windows
        x, ids_restore,mask_indices = self.generate_window_patches(x, rand_left_locations, rand_top_locations, window_size, mask_ratio)
                
        # append the cls tokens at the begining
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        x = self.encoder_pred(x)
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, mask_indices, ids_restore



    def forward_loss(self, imgs, pred, mask_indices,num_window,ids_restore):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = torch.cat([self.patchify(self.sarfeature1(imgs)), self.patchify(self.sarfeature2(imgs)), self.patchify(self.sarfeature3(imgs)), self.patchify(self.sarfeature4(imgs))], dim=-1)
        # print(target.shape)
        # target = self.patchify(imgs)

        N,P,H = target.shape

        target = target.unsqueeze(0).repeat(num_window,1,1,1).view(-1,P,H)

        target = torch.gather(target,dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, target.shape[-1]))

        # if self.norm_pix_loss:
        #     mean = target.mean(dim=-1, keepdim=True)
        #     var = target.var(dim=-1, keepdim=True)
        #     target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch


        loss = (loss * mask_indices).sum() / mask_indices.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, window_size=7, num_window=4,mask_ratio=0.8):
        pred, mask_indices, ids_restore = self.forward_encoder(imgs, window_size,num_window,mask_ratio)
        loss = self.forward_loss(imgs, pred, mask_indices,num_window,ids_restore)
        return loss, pred, mask_indices




def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge448_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=448,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_huge672_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=672,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge996_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=996,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge336_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=336,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_384_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=384,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_448_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=448,
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



def mae_vit_base_patch14_224_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=14, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch8_224_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=8, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

mae_vit_base_patch8_224 = mae_vit_base_patch8_224_dec512d8b
mae_vit_base_patch14_224 = mae_vit_base_patch14_224_dec512d8b
mae_vit_base_patch16_384 = mae_vit_base_patch16_384_dec512d8b
mae_vit_base_patch16_448 = mae_vit_base_patch16_448_dec512d8b

mae_vit_huge336_patch14 = mae_vit_huge336_patch14_dec512d8b
mae_vit_huge448_patch14 = mae_vit_huge448_patch14_dec512d8b
mae_vit_huge672_patch14 = mae_vit_huge672_patch14_dec512d8b
mae_vit_huge996_patch14 =mae_vit_huge996_patch14_dec512d8b





#mae_vit_huge448_patch14 = mae_vit_huge448_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

def vit_tiny(**kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=16, embed_dim=192, depth=12, num_heads=3,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def vit_small(patch_size=16, **kwargs):
    model = MaskedAutoencoderViT(img_size=224,
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model

mae_vit_tiny = vit_tiny
mae_vit_small = vit_small