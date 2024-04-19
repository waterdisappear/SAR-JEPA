import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from tqdm import tqdm
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

CUSTOM_TEMPLATES = {
    'MSTAR_SOC': 'a photo of {}.',
}

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import vision_transformer_irpe

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


def interpolate_pos_embed(model, checkpoint_model):
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

class CustomCLIP(nn.Module):

    def __init__(self, cfg, classnames):
        super().__init__()
        # model = VisionTransformer(
        # img_size=224, patch_size=16, embed_dim=192, in_chans=1, num_classes=len(classnames), depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # model = VisionTransformer(
        # img_size=224, patch_size=16, embed_dim=384, in_chans=1, num_classes=len(classnames), depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6))

        model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=768, in_chans=1, num_classes=len(classnames), depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # model = VisionTransformer(
        # img_size=224, patch_size=16, embed_dim=1024, in_chans=1, num_classes=len(classnames), depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6))


        # model = VisionTransformer(
        # img_size=224, patch_size=14, embed_dim=1280, in_chans=1, num_classes=len(classnames), depth=32, num_heads=16, mlp_ratio=4, qkv_bias=True,
        # norm_layer=partial(nn.LayerNorm, eps=1e-6))
        #
        checkpoint = torch.load('../weights/SAR-JEPA/checkpoint-200.pth',
                                map_location='cpu')

        checkpoint = checkpoint['model']
        checkpoint_model = {k.replace('module.',''):v for k,v in checkpoint.items()}
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # load pre-trained model
        print('load pre-trained model')
        # interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        assert set(msg.missing_keys) == {'head.weight', 'head.bias'}
        print(msg)

        # manually initialize fc layer: following MoCo v3
        from timm.models.layers import trunc_normal_
        trunc_normal_(model.head.weight, std=0.01)

        model.head = torch.nn.Sequential(torch.nn.BatchNorm1d(model.head.in_features, affine=False, eps=1e-6),
                                         model.head)

        # freeze all but the head
        for name, p in model.named_parameters():
            p.requires_grad = False
            # if 'blocks.4' in name:
            #     p.requires_grad_(True)
            # if 'blocks.5' in name:
            #     p.requires_grad_(True)
        for _, p in model.head.named_parameters():
            p.requires_grad = True

        self.image_encoder = model.cuda()


    def forward(self, image):
        # image = torch.concat([image, image, image], 1)
        image_features = self.image_encoder(image)

        return image_features


@TRAINER_REGISTRY.register()
class MIM_linear(TrainerX):
    """ CLIP-Adapter """

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading MAE (backbone: {cfg.MODEL.BACKBONE.NAME})')

        print('Building custom CLIP')
        self.model = CustomCLIP(cfg, classnames)

        print('Turning off gradients in both the image and the text encoder')
        # for name, param in self.model.named_parameters():
        #     if 'adapter' not in name:
        #         param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model, cfg.MODEL.INIT_WEIGHTS)
            # load_pretrained_weights(self.model.image_encoder, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give text_encoder.adapter to the optimizer
        self.optim = build_optimizer(self.model.image_encoder, cfg.OPTIM)
        # self.optim = build_optimizer(self.model.image_encoder, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model('clip', self.model.image_encoder, self.optim, self.sched)

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f'Multiple GPUs detected (n_gpus={device_count}), use all of them!')
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        # loss = F.cross_entropy(output2, label) + self.model.criteria(output1, label)
        loss = F.cross_entropy(output, label)

        self.model_backward_and_update(loss)

        loss_summary = {
            'loss': loss.item(),
            'acc': compute_accuracy(output, label)[0].item()
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch['img']
        label = batch['label']
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print(
                'Note that load_model() is skipped as no pretrained model is given'
            )
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = 'model-best.pth.tar'

        if epoch is not None:
            model_file = 'model.pth.tar-' + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError(
                    'Model not found at "{}"'.format(model_path)
                )

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint['state_dict']
            epoch = checkpoint['epoch']

            # Ignore fixed token vectors
            if 'token_prefix' in state_dict:
                del state_dict['token_prefix']

            if 'token_suffix' in state_dict:
                del state_dict['token_suffix']

            print(
                'Loading weights to {} '
                'from "{}" (epoch = {})'.format(name, model_path, epoch)
            )
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]


    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        # if do_test and self.cfg.TEST.FINAL_MODEL == "best_val":
        #     curr_result = self.test(split="val")
        #     is_best = curr_result > self.best_result
        #     if is_best:
        #         self.best_result = curr_result
        #         self.save_model(
        #             self.epoch,
        #             self.output_dir,
        #             val_result=curr_result,
        #             model_name="model-best.pth.tar"
        #         )

        # if meet_checkpoint_freq or last_epoch:
        #     self.save_model(self.epoch, self.output_dir)