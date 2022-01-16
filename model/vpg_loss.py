#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import torch.nn.functional as F
import torch
import torch.nn as nn
from .custom_util import get_masked_softmax

from fairseq.modules import LayerNorm, MultiheadAttention


class Vpg(nn.Module):
    def __init__(self, embed_dim, args):
        super().__init__()
        self.args = args

        self.target_patch_attn = self.build_target_patch_attention(embed_dim, args)
        self.patch_source_attn = self.build_patch_source_attention(embed_dim, args)

    def build_target_patch_attention(self, embed_dim, args):
        quant_noise = getattr(args, "quant_noise_pq", 0)
        quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=quant_noise,
            qn_block_size=quant_noise_block_size,
        )

    def build_patch_source_attention(self, embed_dim, args):
        quant_noise = getattr(args, "quant_noise_pq", 0)
        quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=quant_noise,
            qn_block_size=quant_noise_block_size,
        )

    def forward(self, target_vec, source_vec, img_vec, encoder_padding_mask):
        target_vec = target_vec
        source_vec = source_vec
        img_vec = img_vec.transpose(0, 1)

        _, target_patch_attn = self.target_patch_attn(
            query=target_vec,
            key=img_vec,
            value=img_vec,
            static_kv=True
        )

        _, source_patch_attn = self.patch_source_attn(
            query=source_vec,
            key=img_vec,
            value=img_vec,
            static_kv=True
        )
        patch_source_attn = source_patch_attn.transpose(2, 1)

        visual_attn_raw = torch.matmul(target_patch_attn, patch_source_attn)
        visual_attn = get_masked_softmax(visual_attn_raw, ~encoder_padding_mask)
        return visual_attn, target_patch_attn


def get_vpg_attn(target_vec, source_vec, img_vec, encoder_padding_mask, is_custom=False, args=None):
    target_vec = target_vec.transpose(0, 1)
    source_vec = source_vec.transpose(0, 1)
    img_vec = img_vec.transpose(2, 1)

    # print(target_vec.shape, source_vec.shape, img_vec.shape)
    # if img_vec.size()[0] != target_vec.size()[0]:
    #     print(target_vec.shape, source_vec.shape, img_vec.shape)

    target_patch_attn = torch.matmul(target_vec, img_vec)
    target_patch_attn = torch.softmax(target_patch_attn, dim=-1)
    source_patch_attn = torch.matmul(source_vec, img_vec)
    source_patch_attn = torch.softmax(source_patch_attn, dim=-1)

    visual_attn_raw = torch.matmul(target_patch_attn, source_patch_attn.transpose(2, 1))
    visual_attn = get_masked_softmax(visual_attn_raw, ~encoder_padding_mask)

    return visual_attn, target_patch_attn


def get_new_vpg_attn(encoder_self_attn, decoder_encoder_attn, encoder_padding_mask, img_len=49):
    encoder_text_len = encoder_padding_mask.size()[-1] - img_len
    encoder_text_padding_mask = encoder_padding_mask[:, :encoder_text_len]

    patch_source_attn = encoder_self_attn[:, encoder_text_len:, :encoder_text_len]
    target_patch_attn = decoder_encoder_attn[:, :, encoder_text_len:]
    visual_attn_raw = torch.matmul(target_patch_attn, patch_source_attn)
    visual_attn = get_masked_softmax(visual_attn_raw, ~encoder_text_padding_mask)

    target_source_direct_attn = decoder_encoder_attn[:, :, :encoder_text_len]
    target_source_direct_attn = get_masked_softmax(target_source_direct_attn, encoder_text_padding_mask)

    return visual_attn, target_patch_attn, target_source_direct_attn


def get_margin_loss(encoder_out, margin, img_len, triplet_loss):
    encoder_out = encoder_out.transpose(0, 1)
    encoder_text_len = encoder_out.size()[-2] - img_len
    text_vec = torch.mean(encoder_out[:, :encoder_text_len, :], dim=-2)
    patch_vec = torch.mean(encoder_out[:, encoder_text_len:, :], dim=-2)

    rand_encoder_out = encoder_out[torch.randperm(encoder_out.size()[0])]
    rand_text_vec = torch.mean(rand_encoder_out[:, :encoder_text_len, :], dim=-2)
    rand_patch_vec = torch.mean(rand_encoder_out[:, encoder_text_len:, :], dim=-2)

    t_output = triplet_loss(text_vec, patch_vec, rand_patch_vec)
    v_output = triplet_loss(patch_vec, text_vec, rand_text_vec)
    output = t_output + v_output
    return output


def get_vpg_dist(net_output, log_probs, copy_type='vpg'):
    x, extra = net_output
    attn, p_gen, src_tokens = extra['attn'][0], extra['p_gen'][0], extra['src_tokens'][0]
    visual_attn, p_visual_copy = extra['visual_attn'][0], extra['p_visual_copy'][0]

    vocab_dist = torch.softmax(x, dim=-1)
    copy_index = src_tokens.unsqueeze(1).expand_as(attn)

    copy_dist = torch.zeros_like(vocab_dist).scatter_add(2, copy_index, attn.type_as(vocab_dist))

    if copy_type in ('vpg', 'single_vpg', 'vpg_test', 'new_vpg', 'new_single_vpg'):
        visual_copy_dist = torch.zeros_like(vocab_dist).scatter_add(2, copy_index, visual_attn.type_as(vocab_dist))
        final_copy_dist = p_visual_copy * visual_copy_dist + (1 - p_visual_copy) * copy_dist

    if copy_type in ('tpg', 'new_tpg'):
        # text pg
        final_dist = (p_gen * vocab_dist) + (1 - p_gen) * copy_dist
        # sys.stdout.write("[get text pg dist]\n")
    elif copy_type in ('single_vpg', 'new_single_vpg'):
        # visual pg
        final_dist = (p_gen * vocab_dist) + (1 - p_gen) * visual_copy_dist
        # sys.stdout.write("[get visual pg dist]\n")
    elif copy_type in ('vpg', 'vpg_test', 'new_vpg'):
        # visual pg
        final_dist = (p_gen * vocab_dist) + (1 - p_gen) * final_copy_dist
        # sys.stdout.write("[get text&visual pg dist]\n")
    else:
        raise NotImplementedError("No Support Copy Type : [] \n".format(copy_type))

    probs = torch.clamp(final_dist, 1e-6, 1 - 1e-6)

    sys.stdout.write("[p_visual_copy]: {}\n".format(p_visual_copy))
    sys.stdout.write("[1 - p_gen]: {}\n".format(1. - p_gen))
    if log_probs:
        return torch.log(probs.float())
    else:
        return probs.float()


if __name__ == '__main__':
    pass
