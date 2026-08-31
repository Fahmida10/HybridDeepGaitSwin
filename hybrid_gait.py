"""HybridDeepGaitSwin — P3D DeepGaitV2 branch + 3D Swin branch + AttentionFusion.

This file is the canonical model definition. Train and eval scripts copy it into
`opengait_toolkit/opengait/modeling/models/hybrid_gait.py` before launch.
Relative imports (`..base_model`, `.swingait`) resolve inside that package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange

from ..base_model import BaseModel
from ..modules import (
    BasicBlock2D,
    BasicBlockP3D,
    HorizontalPoolingPyramid,
    PackSequenceWrapper,
    SeparateBNNecks,
    SeparateFCs,
    SetBlockWrapper,
    conv1x1,
    conv3x3,
)
from .swingait import SwinTransformer3D, adjust_learning_rate, trunc_normal_

blocks_map = {
    '2d': BasicBlock2D,
    'p3d': BasicBlockP3D,
}


class AttentionFusion(nn.Module):
    """Channel-wise attention fusion for two feature maps of equal channel count."""

    def __init__(self, in_channels=512, squeeze_ratio=16):
        super().__init__()
        hidden_dim = max(in_channels // squeeze_ratio, 8)
        self.conv = SetBlockWrapper(
            nn.Sequential(
                conv1x1(in_channels * 2, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv3x3(hidden_dim, hidden_dim),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
                conv1x1(hidden_dim, in_channels * 2),
            )
        )

    def forward(self, deep_feat, swin_feat):
        feats = torch.cat([deep_feat, swin_feat], dim=1)
        score = self.conv(feats)
        score = rearrange(score, 'n (d c) s h w -> n d c s h w', d=2)
        score = F.softmax(score, dim=1)
        return deep_feat * score[:, 0] + swin_feat * score[:, 1]


class HybridDeepGaitSwin(BaseModel):
    """Shared stem + dual branch (DeepGaitV2 P3D + SwinGait transformer) + attention fusion."""

    def __init__(self, cfgs, training):
        self.T_max_iter = cfgs['trainer_cfg'].get('T_max_iter', cfgs['trainer_cfg']['total_iter'])
        super().__init__(cfgs, training=training)

    def build_network(self, model_cfg):
        mode = model_cfg['Backbone']['mode']
        assert mode in blocks_map
        block = blocks_map[mode]

        in_channels = model_cfg['Backbone']['in_channels']
        layers = model_cfg['Backbone']['layers']
        channels = model_cfg['Backbone']['channels']
        swin_layers = model_cfg['Backbone'].get('swin_layers', [1, 1])
        swin_embed_dim = model_cfg['Backbone'].get('swin_embed_dim', 256)
        swin_num_heads = model_cfg['Backbone'].get('swin_num_heads', [16, 32][:len(swin_layers)])
        fusion_cfg = model_cfg.get('Fusion', {})
        self.inference_use_emb2 = model_cfg.get('use_emb2', False)

        strides = [[1, 1], [2, 2], [2, 2], [1, 1]]

        self.inplanes = channels[0]
        self.layer0 = SetBlockWrapper(nn.Sequential(
            conv3x3(in_channels, self.inplanes, 1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True),
        ))
        self.layer1 = SetBlockWrapper(
            self.make_layer(BasicBlock2D, channels[0], strides[0], blocks_num=layers[0], mode='2d')
        )
        self.layer2 = self.make_layer(block, channels[1], strides[1], blocks_num=layers[1], mode=mode)
        self.layer3 = self.make_layer(block, channels[2], strides[2], blocks_num=layers[2], mode=mode)
        self.layer4 = self.make_layer(block, channels[3], strides[3], blocks_num=layers[3], mode=mode)

        self.ulayer = SetBlockWrapper(nn.UpsamplingBilinear2d(size=(30, 20)))
        self.transformer = SwinTransformer3D(
            patch_size=[1, 2, 2],
            in_chans=channels[1],
            embed_dim=swin_embed_dim,
            depths=list(swin_layers),
            num_heads=list(swin_num_heads),
            window_size=[3, 3, 5],
            downsample=[1, 0] if len(swin_layers) > 1 else [0],
            drop_path_rate=0.1,
            patch_norm=True,
        )

        swin_out_channels = swin_embed_dim * (2 if len(swin_layers) > 1 else 1)
        if swin_out_channels != channels[3]:
            self.swin_proj = SetBlockWrapper(
                nn.Sequential(
                    conv1x1(swin_out_channels, channels[3]),
                    nn.BatchNorm2d(channels[3]),
                    nn.ReLU(inplace=True),
                )
            )
        else:
            self.swin_proj = nn.Identity()

        self.fusion = AttentionFusion(
            in_channels=channels[3],
            squeeze_ratio=fusion_cfg.get('squeeze_ratio', 16),
        )

        bin_num = model_cfg.get('bin_num', [16])
        parts_num = bin_num[0]

        self.FCs = SeparateFCs(parts_num, channels[3], channels[2])
        self.BNNecks = SeparateBNNecks(
            parts_num,
            channels[2],
            class_num=model_cfg['SeparateBNNecks']['class_num'],
        )
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=bin_num)

    def make_layer(self, block, planes, stride, blocks_num, mode='p3d'):
        if max(stride) > 1 or self.inplanes != planes * block.expansion:
            if mode == 'p3d':
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes, planes * block.expansion,
                        kernel_size=[1, 1, 1], stride=[1, *stride],
                        padding=[0, 0, 0], bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride=stride),
                    nn.BatchNorm2d(planes * block.expansion),
                )
        else:
            downsample = lambda x: x

        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        step = [1, 1]
        for _ in range(1, blocks_num):
            layers.append(block(self.inplanes, planes, stride=step))
        return nn.Sequential(*layers)

    def _align_spatial(self, src, ref):
        if src.shape[-2:] == ref.shape[-2:]:
            return src
        n, c, s, _, _ = src.shape
        src = rearrange(src, 'n c s h w -> (n s) c h w')
        src = F.interpolate(src, size=ref.shape[-2:], mode='bilinear', align_corners=False)
        return rearrange(src, '(n s) c h w -> n c s h w', n=n, s=s)

    def get_optimizer(self, optimizer_cfg):
        self.msg_mgr.log_info(optimizer_cfg)
        from utils import get_valid_args, get_attr_from

        optimizer_cls = get_attr_from([optim], optimizer_cfg['solver'])
        valid_arg = get_valid_args(optimizer_cls, optimizer_cfg, ['solver'])

        transformer_no_decay = ['patch_embed', 'norm', 'relative_position_bias_table']
        transformer_params = list(self.transformer.named_parameters())
        params_list = [
            {
                'params': [p for n, p in transformer_params if any(nd in n for nd in transformer_no_decay)],
                'lr': optimizer_cfg['lr'],
                'weight_decay': 0.0,
            },
            {
                'params': [p for n, p in transformer_params if not any(nd in n for nd in transformer_no_decay)],
                'lr': optimizer_cfg['lr'],
                'weight_decay': optimizer_cfg['weight_decay'],
            },
            {
                'params': list(self.fusion.parameters()) + list(self.swin_proj.parameters()),
                'lr': optimizer_cfg['lr'] * 0.5,
                'weight_decay': optimizer_cfg['weight_decay'],
            },
            {
                'params': self.FCs.parameters(),
                'lr': optimizer_cfg['lr'] * 0.1,
                'weight_decay': optimizer_cfg['weight_decay'],
            },
            {
                'params': self.BNNecks.parameters(),
                'lr': optimizer_cfg['lr'] * 0.1,
                'weight_decay': optimizer_cfg['weight_decay'],
            },
        ]
        for i in range(5):
            if hasattr(self, f'layer{i}'):
                params_list.append({
                    'params': getattr(self, f'layer{i}').parameters(),
                    'lr': optimizer_cfg['lr'] * 0.1,
                    'weight_decay': optimizer_cfg['weight_decay'],
                })

        optimizer = optimizer_cls(params_list, **valid_arg)
        for group in optimizer.param_groups:
            group['initial_lr'] = group['lr']
        return optimizer

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                if m.affine:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                    nn.init.constant_(m.bias.data, 0.0)

    def forward(self, inputs):
        if self.training and hasattr(self, 'optimizer'):
            adjust_learning_rate(self.optimizer, self.iteration, T_max_iter=self.T_max_iter)

        ipts, labs, _, _, seqL = inputs

        if len(ipts[0].size()) == 4:
            sils = ipts[0].unsqueeze(1)
        else:
            sils = ipts[0].transpose(1, 2).contiguous()
        del ipts

        out0 = self.layer0(sils)
        out1 = self.layer1(out0)
        out2 = self.layer2(out1)

        swin_feat = self.ulayer(out2)
        swin_feat = self.transformer(swin_feat)
        swin_feat = self.swin_proj(swin_feat)

        deep_feat = self.layer4(self.layer3(out2))
        swin_feat = self._align_spatial(swin_feat, deep_feat)

        fused = self.fusion(deep_feat, swin_feat)

        outs = self.TP(fused, seqL, options={'dim': 2})[0]
        feat = self.HPP(outs)
        embed_1 = self.FCs(feat)
        embed_2, logits = self.BNNecks(embed_1)

        embed = embed_2 if self.inference_use_emb2 else embed_1

        return {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs},
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w'),
            },
            'inference_feat': {
                'embeddings': embed,
            },
        }
