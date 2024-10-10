# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin
from torch.nn.modules.loss import _Loss

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_config as configs

from .vit_resnet_skip import ResNetV2


logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def hid2fea(hidden_states, reshape_scale):
    B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch=60, hidden) to (B, h, w, hidden)
    # h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
    h, w = reshape_scale
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    return x


def dice_coefficient(outputs, targets, channel=1, threshold=0.5, eps=1e-8):
    outputs = torch.sigmoid(outputs)
    # if channel == 1:
    y_pred, y_truth = outputs, targets
    y_pred = y_pred > threshold
    # y_pred直接是一个boolen值
    intersection = torch.sum(torch.mul(y_pred, y_truth)) + eps / 2
    union = torch.sum(y_pred) + torch.sum(y_truth) + eps
    dice = 2 * intersection / union
    # print("targets, intersection, union:", torch.unique(targets, return_counts=True), intersection, union)
    return dice


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        y_pred = torch.sigmoid(y_pred)
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps
        dice = 2 * intersection / union
        dice_loss = 1 - dice

        return dice_loss
    

class ClassificationHead(nn.Module):
    def __init__(self, cfg, spatial=False):
        super(ClassificationHead, self).__init__()
        self.config = cfg
        self.spatial = spatial
        self.cls = cfg.n_classes
        self.cls_loss = nn.L1Loss()
        if self.cls == 1:    
            self.cls += 1                       
        self.to_latent = nn.Identity()
        self.cls_head = nn.Sequential(
            nn.Linear(cfg.channel, 512),
            nn.LayerNorm(512),
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.Linear(64, self.cls)
        )
        
    def forward(self, x, label):
        x = x.mean(dim = 1) 
        x = self.to_latent(x)
        x = self.cls_head(x)
        pred = F.softmax(x, dim=1)
        
        pred = torch.squeeze(torch.squeeze(pred, dim=-1), dim=-1)
        proposition = self.cal_area(label, pred)
        proposition = proposition.to(x.device)
        loss = self.cls_loss(pred, proposition)
        return loss


    def cal_area(self, label):
        temp_label = label.clone()
        prop = []
        B, C, _, _ = label.shape
        ar, area = torch.unique(label, return_counts=True)
        if C > 1:
            for i in range(C):
                temp = temp_label[:, i, :, :]
                temp[torch.where(temp == 1)] = i + 1
                temp = torch.unsqueeze(temp, dim=1)
                temp_label[:, i:i + 1, :, :] = temp
            temp_label = torch.sum(temp_label, dim=1)
        else:
            temp_label = torch.squeeze(temp_label, dim=1)

        for j in range(B):
            num = len(torch.unique(temp_label[j, :, :]))
            ar, area = torch.unique(temp_label[j, :, :], return_counts=True)
            proposition = torch.zeros([1, self.cls+1])
            whole_area = torch.sum(area)
            for i in range(num):
                cls = int(ar[i])
                proposition[0, cls] = area[i] / whole_area
            if self.cls > 1:
                proposition = proposition[0, 1:] / torch.sum(proposition[0, 1:])
                proposition = torch.unsqueeze(proposition, dim=0)
            prop.append(proposition)
        prop = torch.cat(prop, dim=0)
        return prop




def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)

def hid2fea(hidden_states, scale):
    B, n_patch, hidden = hidden_states.size()  
    h, w = scale
    x = hidden_states.permute(0, 2, 1)
    x = x.contiguous().view(B, hidden, h, w)
    return x


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)


    def forward(self, q, hidden_states):
        if q is not None:
            mixed_query_layer = q
        else:
            mixed_query_layer = self.query(hidden_states)

        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        reception_field = context_layer
        #print("reception_field:", reception_field.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        # 模型中的A
        return attention_output, weights, reception_field


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, slot=False, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = True
        self.config = config
        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1]) 
            self.scale = (img_size[0] // patch_size_real[0]), (img_size[1] // patch_size_real[1]) 
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False
        
        if slot:
            self.slot_encoder = ResNetV2(channel=2, block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor).cuda()
            in_channels = self.slot_encoder.width * 16
        else:
            if self.hybrid:
                self.hybrid_model = ResNetV2(channel=3, block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor).cuda()
                in_channels = self.hybrid_model.width * 16

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size).cuda()
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size)).cuda()

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, slot=False):
        if slot:
            x, _ = self.slot_encoder(x)
            features = None
        else:
            if x.size()[1] == 1:
                x = x.repeat(1,3,1,1)
            if self.hybrid:
                x, features = self.hybrid_model(x)
            else:
                features = None
 
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)   
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)

        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, query, x):
        current_slice = x
        x = self.attention_norm(x)
        x, weights, reception_field = self.attn(query, x)
        A = x + current_slice

        h = A
        A = self.ffn_norm(A)
        A = self.ffn(A)
        Attention_matrix = A + h

        return Attention_matrix, weights, reception_field

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


## 12-based transformer encoder
class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()

        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.Update_Q = Mlp(config)
        #####
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = Linear(config.hidden_size, self.all_head_size)
        #####
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    # 0710 todo: 
    def forward(self, query, hidden_states):
        ###
        device = hidden_states.device
        current_slice = hidden_states
        # first slice: query == 0
        if query is not None:   
            current_query = query
        else:
            query = self.query(hidden_states)
            current_query = query
        ###
        attn_weights = []
        k = 0
        for layer_block in self.layer:
            if k == 0:
                # query是传进去的
                hidden_states, weights, reception_field = layer_block(query, hidden_states)
            else:
                query = None
                hidden_states, weights, reception_field = layer_block(query, hidden_states)
            k += 1

        velocity_field = self.encoder_norm(hidden_states)
        update_query_ori = velocity_field + current_query
        update_query = self.ffn_norm(update_query_ori)
        update_query = self.Update_Q(update_query)
        Attention_matrix = current_slice + update_query
        return Attention_matrix, attn_weights, update_query, update_query_ori


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=False)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        # print("x_de:", x.shape)
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        reshape_scale = self.config.rescale
        x = hid2fea(hidden_states, reshape_scale)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x



class VisionTransformer(nn.Module):
    def __init__(self, config, num_classes, img_size=224, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.encoder = Encoder(config, vis)
        self.clshead = ClassificationHead(config, num_classes)
        self.decoder = DecoderCup(config)
        self.seg_loss = SoftDiceLoss()
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, query, embedding_output, features, label, state):
        hidden, attn_weights, update_query, update_query_ori = self.encoder(query, embedding_output) 
        slice = self.decoder(hidden, features)
        logits = self.segmentation_head(slice)
        if state == 'test':
            return logits, update_query
        else:
            seg_loss = self.seg_loss(logits, label)
            return logits, update_query, seg_loss, update_query_ori


    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


class S2S_TransNet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2):
        super(S2S_TransNet, self).__init__()
        self.encoder = Embeddings(config, img_size=img_size)
        self.backbone = VisionTransformer(config, num_classes=num_classes, img_size=img_size)
        self.config = config
        self.img_size = img_size
        self.num_classes = num_classes

    def forward(self, data3D, label3D=None, state='test'):
        _, _, _, _, D = data3D.shape
        slots_predicted_list, logit_predicted_list = [], []
        seg_all = 0

        for d in range(D):
            if d == 0:
                query = None
            else:
                query = predicted_slots

            slice = data3D[:, :, :, :, d]
            embedding_output, features = self.encoder(slice)

            if state =='train':
                label = label3D[:, :, :, :, d]
                seg_logit, predicted_slots, seg_loss, _ = self.backbone(query, embedding_output, features, label, state)
                seg_all += seg_loss
                seg_logit = seg_logit.unsqueeze(dim=-1)
                memory_slot = predicted_slots.unsqueeze(dim=-1)
                slots_predicted_list.append(memory_slot)
                logit_predicted_list.append(seg_logit)
            elif state == 'test':
                seg_logit, predicted_slots = self.backbone(query, embedding_output, features, label3D, state)
                seg_logit = seg_logit.unsqueeze(dim=-1)
                slots_saved = predicted_slots.unsqueeze(dim=-1)
                logit_predicted_list.append(seg_logit)
                slots_predicted_list.append(slots_saved)

        if state == 'train':
            outputs_pred = torch.cat(slots_predicted_list, dim=-1)
            outputs = torch.cat(logit_predicted_list, dim=-1)
            return {
                "slot_pred": outputs_pred,
                "outputs": outputs,
                "seg_loss": seg_all,
            }
        elif state == 'test':
            outputs = torch.cat(logit_predicted_list, dim=-1)
            return outputs

CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

