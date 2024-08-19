
from functools import partial
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from torch.autograd import Variable
import vision_transformer_irpe
import torch
import torch.nn as nn
import zipfile
from io import BytesIO
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests

from PIL import Image
from sklearn.preprocessing import MinMaxScaler


RESOLUTION = 224
PATCH_SIZE = 16
GITHUB_RELEASE = "https://github.com/sayakpaul/probing-vits/releases/download/v1.0.0/probing_vits.zip"
FNAME = "probing_vits.zip"
MODELS_ZIP = {
    "vit_dino_base16": "Probing_ViTs/vit_dino_base16.zip",
    "vit_b16_patch16_224": "Probing_ViTs/vit_b16_patch16_224.zip",
    "vit_b16_patch16_224-i1k_pretrained": "Probing_ViTs/vit_b16_patch16_224-i1k_pretrained.zip",
}


# Copyright 2022 Google LLC.
# SPDX-License-Identifier: Apache-2.0
# Author: Maithra Raghu <maithra@google.com>

def compute_distance_matrix(patch_size, num_patches, length):
    """compute_distance_matrix: Computes the distance matrix for the patches in the image

    Args:
        patch_size (int): the size of the patch
        num_patches (int): the number of patches in the image
        length (int): the length of the image

    Returns:
        distance_matrix (np.ndarray): The distance matrix for the patches in the image
    """
    distance_matrix = np.zeros((num_patches, num_patches))
    for i in range(num_patches):
        for j in range(num_patches):
            if i == j:  # zero distance
                continue

            xi, yi = (int(i / length)), (i % length)
            xj, yj = (int(j / length)), (j % length)
            distance_matrix[i, j] = patch_size * np.linalg.norm([xi - xj, yi - yj])

    return distance_matrix


def compute_mean_attention_dist(patch_size, attention_weights, num_cls_tokens=1):
    """compute_mean_attention_dist: Computes the mean attention distance for the image

    Args:
        patch_size (int): the size of the patch
        attention_weights (np.ndarray): The attention weights for the image
        num_cls_tokens (int, optional): The number of class tokens. Defaults to 1.

    Returns:
        mean_distances (np.ndarray): The mean attention distance for the image
    """
    # The attention_weights shape = (batch, num_heads, num_patches, num_patches)
    attention_weights = attention_weights[
        ..., num_cls_tokens:, num_cls_tokens:
    ]  # Removing the CLS token
    num_patches = attention_weights.shape[-1]
    length = int(np.sqrt(num_patches))
    assert length ** 2 == num_patches, "Num patches is not perfect square"

    distance_matrix = compute_distance_matrix(patch_size, num_patches, length)
    h, w = distance_matrix.shape

    distance_matrix = distance_matrix.reshape((1, 1, h, w))
    # The attention_weights along the last axis adds to 1
    # this is due to the fact that they are softmax of the raw logits
    # summation of the (attention_weights * distance_matrix)
    # should result in an average distance per token
    mean_distances = attention_weights * distance_matrix
    mean_distances = np.sum(
        mean_distances, axis=-1
    )  # sum along last axis to get average distance per token
    mean_distances = np.mean(
        mean_distances, axis=-1
    )  # now average across all the tokens

    return mean_distances


def rollout(attentions, discard_ratio, head_fusion, patch_size=16, img_size=224, column_index=None, row_index=None, depth=11, color='tab:red'):
    """rollout: rollout the attention map based on the attentions

    Args:
        attentions (tensor): attentions get from the model
        discard_ratio (float): the ratio of the lowest attentions to be discarded
        head_fusion (str): the way to fuse the attentions of different heads
        patch_size (int, optional): the size of patch. Defaults to 16.
        img_size (int, optional): the size of image. Defaults to 224.
        column_index (int, optional): patch column index. Defaults to None.
        row_index (int, optional): patch row index. Defaults to None.
        depth (int, optional): the depth of the model. Defaults to 11.
        color (str, optional): the color of the rectangle. Defaults to 'tab:red'.

    Returns:
        mask (np.ndarray): the attention map after rollout
        rect (patches.Rectangle): the rectangle of the patch
    """
    num_patches = img_size // patch_size
    result = torch.eye(attentions[0].size(-1)) # (197, 197)
    with torch.no_grad():
        for attention_len in range(len(attentions)):
            attention = attentions[attention_len]
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1) # (1, 197×197)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, False)
            indices = indices[indices != 0]
            flat[0, indices] = 0

            I = torch.eye(attention_heads_fused.size(-1)) # (197, 197)
            a = (attention_heads_fused + 1.0 * I) / 2 # (1, 197, 197)
            a = a / a.sum(dim=-1).double()

            result = torch.matmul(a, result.double())


            if attention_len == depth:
                break

    if column_index == None or row_index == None:
        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0, 0, 1:]
        rect = None
    else:
        mask = result[0, (row_index-1) * num_patches + column_index, 1:]
        rect = patches.Rectangle((num_patches * (column_index - 1), num_patches * (row_index - 1)), num_patches, num_patches, linewidth=3,
                                 edgecolor=color,
                                 facecolor='none')
    # In case of 224x224 image, this brings us from 196 to 14
    mask = mask.reshape(num_patches, num_patches).numpy()
    mask = mask / np.max(mask)

    return mask, rect


class ViTAttentionGetWithGrad:
    """ViTAttentionGetWithGrad: get the attention map and the gradients of the attention map

    Args:
        model (nn.Module): the model
        attention_layer_name (str, optional): the name of the attention layer. Defaults to 'attn_drop'.
        discard_ratio (float, optional): the ratio of the lowest attentions to be discarded. Defaults to 0.9.
    """
    def __init__(self, model, attention_layer_name='attn_drop',
        discard_ratio=0.9):
        self.model = model
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []
        self.attention_gradients = []

    def get_attention(self, module, input, output):
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor).cpu()

        return self.attentions


class VisionTransformer(vision_transformer_irpe.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """

    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool

        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome




if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, in_chans=1, num_classes=10, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    path = 'D:\MIM\weight_101k\\vit_b\LoMaR_our\\checkpoint-200.pth'
    # path = 'D:\MIM\weight_101k\\vit_b\SAR_HOG\\checkpoint-200.pth'
    path = 'D:\MIM_weight\weight_101k\\vit_b\SAR_tidu\\checkpoint-200.pth'

    print(path.split('\\'))
    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = checkpoint['model']
    checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    state_dict = net.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    # load pre-trained model
    print('load pre-trained model')
    # interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = net.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    print(msg)

    data = cv2.imread('./data/HB19827.jpeg', cv2.IMREAD_GRAYSCALE)
    data = cv2.imread('./data/A3_692.png', cv2.IMREAD_GRAYSCALE)
    transf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
    test_image_tensor = transf(data)
    test_image_tensor = test_image_tensor.unsqueeze(0)
    # print(test_image_tensor.shape)

    net.eval()

    outputs = net(test_image_tensor)
    attns_get_with_grad = ViTAttentionGetWithGrad(net)
    attention_score = attns_get_with_grad(test_image_tensor)

    # print(attns[0].shape)

    # Build the mean distances for every Transformer block.
    mean_distances = {
        f"{name}_mean_dist": compute_mean_attention_dist(
            patch_size=PATCH_SIZE,
            attention_weights=attention_weight.detach().numpy(),
        )
        for name, attention_weight in enumerate(attention_score)
    }

    # Get the number of heads from the mean distance output.
    num_heads = mean_distances["0_mean_dist"].shape[-1]

    # Print the shapes
    print(f"Num Heads: {num_heads}.")

    plt.figure(figsize=(9, 9))

    for idx in range(len(mean_distances)):
        mean_distance = mean_distances[f"{idx}_mean_dist"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        plt.scatter(x=x, y=y, label=f"transformer_block_{idx}")

    # plt.legend(loc="lower right")
    plt.xlabel("Attention Head", fontsize=14)
    plt.ylabel("Attention Distance", fontsize=14)
    plt.title("vit_base_i21k_patch16_224", fontsize=14)
    plt.savefig(path.split('\\')[-2]+".pdf", dpi=600, bbox_inches='tight')
    plt.grid()
    plt.show()