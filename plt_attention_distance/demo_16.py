import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2
from tqdm import *
config = {
    "font.family": 'Times New Roman',
    "font.sans-serif": 'Times New Roman',
    "font.size": 10.5,
    "mathtext.fontset": 'stix',
    "font.serif": 'Times New Roman',
}
rcParams.update(config)
def att_MAE(path, data):
    from demo_MAE import torch, VisionTransformer, partial, nn, transforms, ViTAttentionGetWithGrad, compute_mean_attention_dist, PATCH_SIZE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, in_chans=1, num_classes=10, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = checkpoint['model']
    checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    state_dict = net.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            # print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    # load pre-trained model
    # print('load pre-trained model')
    # interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = net.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    # print(msg)

    transf = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
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



    return mean_distances
    # Print the shapes

def att_lomar(path, data):
    from demo_LoMaR import torch, VisionTransformer, partial, nn, transforms, ViTAttentionGetWithGrad, compute_mean_attention_dist, PATCH_SIZE

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, in_chans=1, num_classes=10, depth=12, num_heads=12,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

    checkpoint = torch.load(path, map_location='cpu')
    checkpoint = checkpoint['model']
    checkpoint_model = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    state_dict = net.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            # print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    # load pre-trained model
    # print('load pre-trained model')
    # interpolate_pos_embed(model, checkpoint_model)

    # load pre-trained model
    msg = net.load_state_dict(checkpoint_model, strict=False)
    assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
    # print(msg)

    transf = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
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



    return mean_distances
    # Print the shapes


fig = plt.figure(figsize=(12,4))
data = cv2.imread('./data/HB19827.jpeg', cv2.IMREAD_GRAYSCALE)


import os
path = "./data"
files= os.listdir(path)

# data = cv2.imread('./data/Ship_C04S02N0767.jpg', cv2.IMREAD_GRAYSCALE)
temp = np.zeros((12,12))
for file in tqdm(files):
    data = cv2.imread('./data/'+file, cv2.IMREAD_GRAYSCALE)
    path = 'D:\MIM_weight\weight_101k\\vit_b\MAE\\checkpoint-200.pth'
    mean_distances = att_MAE(path, data)
    # Get the number of heads from the mean distance output.
    num_heads = mean_distances["0_mean_dist"].shape[-1]
    # print(f"Num Heads: {num_heads}.")
    for idx in range(len(mean_distances)):
        mean_distance = mean_distances[f"{idx}_mean_dist"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        temp[idx, :] = temp[idx, :] + y

ax = plt.subplot(141)
for idx in range(len(mean_distances)):
    mean_distance = mean_distances[f"{idx}_mean_dist"]
    x = [idx] * num_heads
    plt.scatter(x=x, y=temp[idx, :]/16, label=f"transformer_block_{idx}")
# plt.legend(loc="lower right")
plt.xlabel("(a) MAE")
plt.ylabel("Attention Distance (Pixels)")
# plt.title("MAE", fontsize=12)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)1
plt.xticks(np.arange(0, 12, 1), [f'{i}' for i in range(1, 13, 1)])
ax.tick_params("both", direction='in') #"y", 'x', 'both'
plt.grid(axis='y', linestyle='--')
plt.grid(axis='x', linestyle='--')


temp = np.zeros((12,12))
for file in tqdm(files):
    data = cv2.imread('./data/'+file, cv2.IMREAD_GRAYSCALE)
    path = 'D:\MIM_weight\weight_101k\\vit_b\LoMaR_our\\checkpoint-200.pth'
    mean_distances = att_lomar(path, data)
    # Get the number of heads from the mean distance output.
    num_heads = mean_distances["0_mean_dist"].shape[-1]
    # print(f"Num Heads: {num_heads}.")
    for idx in range(len(mean_distances)):
        mean_distance = mean_distances[f"{idx}_mean_dist"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        temp[idx, :] = temp[idx, :] + y

ax = plt.subplot(142)
for idx in range(len(mean_distances)):
    mean_distance = mean_distances[f"{idx}_mean_dist"]
    x = [idx] * num_heads
    plt.scatter(x=x, y=temp[idx, :]/16, label=f"transformer_block_{idx}")
# plt.legend(loc="lower right")
plt.xlabel("(b) LoMaR-SAR")
# plt.ylabel("Attention Distance (Pixels)")
# plt.title("LoMaR", fontsize=12)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)1
plt.xticks(np.arange(0, 12, 1), [f'{i}' for i in range(1, 13, 1)])
ax.tick_params("both", direction='in') #"y", 'x', 'both'
plt.grid(axis='y', linestyle='--')  # 生成网格
plt.grid(axis='x', linestyle='--')  # 生成网格

temp = np.zeros((12,12))
for file in tqdm(files):
    data = cv2.imread('./data/'+file, cv2.IMREAD_GRAYSCALE)
    path = 'D:\MIM_weight\weight_101k\\vit_b\\reconstruction\\checkpoint-200.pth'
    mean_distances = att_lomar(path, data)
    # Get the number of heads from the mean distance output.
    num_heads = mean_distances["0_mean_dist"].shape[-1]
    # print(f"Num Heads: {num_heads}.")
    for idx in range(len(mean_distances)):
        mean_distance = mean_distances[f"{idx}_mean_dist"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        temp[idx, :] = temp[idx, :] + y

ax = plt.subplot(143)
for idx in range(len(mean_distances)):
    mean_distance = mean_distances[f"{idx}_mean_dist"]
    x = [idx] * num_heads
    plt.scatter(x=x, y=temp[idx, :]/16, label=f"transformer_block_{idx}")    
# plt.legend(loc="lower right")
plt.xlabel("(c) PGCA")
# plt.ylabel("Attention Distance (Pixels)")
# plt.title("PGCA", fontsize=12)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)1
plt.xticks(np.arange(0, 12, 1), [f'{i}' for i in range(1, 13, 1)])
ax.tick_params("both", direction='in') #"y", 'x', 'both'
plt.grid(axis='y', linestyle='--')
plt.grid(axis='x', linestyle='--')


temp = np.zeros((12,12))
for file in tqdm(files):
    data = cv2.imread('./data/'+file, cv2.IMREAD_GRAYSCALE)
    path = 'D:\MIM_weight\weight_101k\\vit_b\SAR_tidu\\checkpoint-200.pth'
    mean_distances = att_lomar(path, data)
    # Get the number of heads from the mean distance output.
    num_heads = mean_distances["0_mean_dist"].shape[-1]
    # print(f"Num Heads: {num_heads}.")
    for idx in range(len(mean_distances)):
        mean_distance = mean_distances[f"{idx}_mean_dist"]
        x = [idx] * num_heads
        y = mean_distance[0, :]
        temp[idx, :] = temp[idx, :] + y

ax = plt.subplot(144)
for idx in range(len(mean_distances)):
    mean_distance = mean_distances[f"{idx}_mean_dist"]
    x = [idx] * num_heads
    plt.scatter(x=x, y=temp[idx, :]/16, label=f"transformer_block_{idx}")    
    
# plt.legend(loc="lower right")
plt.xlabel("(d) Ours")
# plt.ylabel("Attention Distance (Pixels)")
# plt.title("Our", fontsize=12)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)1
plt.xticks(np.arange(0, 12, 1), [f'{i}' for i in range(1, 13, 1)])
ax.tick_params("both", direction='in') #"y", 'x', 'both'
plt.grid(axis='y', linestyle='--')
plt.grid(axis='x', linestyle='--')



plt.savefig("./attention_distance.pdf", dpi=600, bbox_inches='tight')
# plt.tight_layout()
plt.show()
