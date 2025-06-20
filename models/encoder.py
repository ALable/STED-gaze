# --------------------------------------------------------
# Copyright (C) 2019 NVIDIA Corporation. All rights reserved.
# NVIDIA Source Code License (1-Way Commercial)
# Code written by Seonwook Park, Shalini De Mello
# --------------------------------------------------------
# Code modified by Yufeng Zheng
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np

from .densenet import (
    DenseNetInitialLayers,
    DenseNetBlock,
    DenseNetTransitionDown,
)
from core import DefaultConfig
config = DefaultConfig()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class Encoder(nn.Module):

    def __init__(self, num_all_pseudo_labels, num_all_embedding_features, configuration):
        super(Encoder, self).__init__()
        self.configuration = configuration
        self.encoder = DenseNetEncoder(num_blocks=config.densenet_blocks)
        c_now = list(self.children())[-1].c_now

        # self.encoder_fc_pseudo_labels1 = nn.Linear(c_now * 16, 96)
        self.encoder_fc_pseudo_labels1 = nn.Linear(c_now*4, 96)
        self.encoder_fc_pseudo_labels2 = nn.Linear(96, int(num_all_pseudo_labels))
        self.tanh = nn.Tanh()
        self.leakyrelu = nn.LeakyReLU()
        self.encoder_fc_embeddings1 = nn.Linear(c_now * 4, c_now * 4)
        # self.encoder_fc_embeddings1 = nn.Linear(c_now * 16, c_now * 4)
        self.encoder_fc_embeddings2 = nn.Linear(c_now * 4, int(num_all_embedding_features))

        self.encoder_fc_pseudo_labels2.weight.data.fill_(0)
        self.encoder_fc_pseudo_labels2.bias.data.fill_(0)
        '''
        self.encoder_fc_pseudo_labels = nn.Linear(c_now * 4, int(num_all_pseudo_labels))
        self.tanh = nn.Tanh()
        self.encoder_fc_embeddings = nn.Linear(c_now * 4, int(num_all_embedding_features))
        '''

    def forward(self, image):
        x = self.encoder(image)
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        # Create latent codes
        #flat_pseudo_labels = 0.5 * np.pi * self.tanh(self.encoder_fc_pseudo_labels(x))
        #flat_embeddings = self.encoder_fc_embeddings(x)
        flat_pseudo_labels = 0.5 * np.pi * self.tanh(self.encoder_fc_pseudo_labels2(
            self.leakyrelu(self.encoder_fc_pseudo_labels1(x))))
        flat_embeddings = self.encoder_fc_embeddings2(self.leakyrelu(self.encoder_fc_embeddings1(x)))
        # Split the pseudo labels and embeddings up
        pseudo_labels = []
        idx_pl = 0
        for dof, num_feats in self.configuration:
            if dof == 0:
                pseudo_label = None
            else:
                pseudo_label = flat_pseudo_labels[:, idx_pl:(idx_pl + dof)]
            pseudo_labels.append(pseudo_label)
            idx_pl += dof

        embeddings = []
        idx_e = 0
        for dof, num_feats in self.configuration:
            len_embedding = (dof + 1) * num_feats
            flat_embedding = flat_embeddings[:, idx_e:(idx_e + len_embedding)]
            embedding = flat_embedding.reshape(-1, dof + 1, num_feats)
            # embedding = nn.functional.normalize(embedding, dim=2)
            embeddings.append(embedding)
            idx_e += len_embedding
        return [pseudo_labels, embeddings]
# class Encoder(nn.Module):
#     def __init__(self, num_all_pseudo_labels, num_all_embedding_features, configuration):
#         super(Encoder, self).__init__()
#         self.configuration = configuration
#         self.encoder = DenseNetEncoder(num_blocks=config.densenet_blocks)
#         c_now = self.encoder.c_now
#         self.leakyrelu = nn.LeakyReLU()
#         self.tanh = nn.Tanh()
#         # First, generate the full embedding vector
#         self.fc_embed1 = nn.Linear(c_now * 4, c_now * 4)
#         self.fc_embed2 = nn.Linear(c_now * 4, int(num_all_embedding_features))

#         # For each embedding part, create independent pseudo_label layers
#         self.pseudo_fc1_list = nn.ModuleList()
#         self.pseudo_fc2_list = nn.ModuleList()
#         self.pseudo_label_dims = []
#         self.embedding_dims = []
#         idx_e = 0
#         for dof, num_feats in self.configuration:
#             len_embedding = (dof + 1) * num_feats
#             self.embedding_dims.append(len_embedding)
#             if dof == 0:
#                 self.pseudo_fc1_list.append(None)
#                 self.pseudo_fc2_list.append(None)
#                 self.pseudo_label_dims.append(0)
#             else:
#                 # Each branch has its own FC layers
#                 self.pseudo_fc1_list.append(nn.Linear(len_embedding, 96))
#                 fc2 = nn.Linear(96, dof)
#                 fc2.weight.data.fill_(0)
#                 fc2.bias.data.fill_(0)
#                 self.pseudo_fc2_list.append(fc2)
#                 self.pseudo_label_dims.append(dof)
#             idx_e += len_embedding

#     def forward(self, image):
#         x = self.encoder(image)
#         batch_size = x.shape[0]
#         x = x.view(batch_size, -1)

#         # 1. Generate the full embedding vector
#         embed_mid = self.leakyrelu(self.fc_embed1(x))
#         flat_embeddings = self.fc_embed2(embed_mid)

#         # 2. Split embedding into parts according to configuration
#         embeddings = []
#         idx_e = 0
#         for dof, num_feats in self.configuration:
#             len_embedding = (dof + 1) * num_feats
#             flat_embedding = flat_embeddings[:, idx_e:(idx_e + len_embedding)]
#             embedding = flat_embedding.reshape(-1, dof + 1, num_feats)
#             embeddings.append(embedding)
#             idx_e += len_embedding

#         # 3. For each embedding part, generate the corresponding pseudo_label
#         pseudo_labels = []
#         for i, (dof, num_feats) in enumerate(self.configuration):
#             if dof == 0:
#                 pseudo_label = None
#             else:
#                 # Flatten the embedding part for FC input
#                 emb_part = embeddings[i].reshape(batch_size, -1)
#                 pseudo_mid = self.leakyrelu(self.pseudo_fc1_list[i](emb_part))
#                 pseudo_label = 0.5 * np.pi * self.tanh(self.pseudo_fc2_list[i](pseudo_mid))
#             pseudo_labels.append(pseudo_label)

#         return [pseudo_labels, embeddings]

class DenseNetEncoder(nn.Module):

    def __init__(self, num_blocks=4, num_layers_per_block=4,
                 p_dropout=0.0, compression_factor=1.0,
                 activation_fn=nn.LeakyReLU, normalization_fn=nn.InstanceNorm2d):
        super(DenseNetEncoder, self).__init__()
        self.c_at_end_of_each_scale = []

        # Initial down-sampling conv layers
        self.initial = DenseNetInitialLayers(growth_rate=config.growth_rate,
                                             activation_fn=activation_fn,
                                             normalization_fn=normalization_fn)
        c_now = list(self.children())[-1].c_now
        self.c_at_end_of_each_scale.append(list(self.children())[-1].c_list[0])

        assert (num_layers_per_block % 2) == 0
        for i in range(num_blocks):
            i_ = i + 1
            # Define dense block
            self.add_module('block%d' % i_, DenseNetBlock(
                c_now,
                num_layers=num_layers_per_block,
                growth_rate=config.growth_rate,
                p_dropout=p_dropout,
                activation_fn=activation_fn,
                normalization_fn=normalization_fn,
            ))
            c_now = list(self.children())[-1].c_now
            self.c_at_end_of_each_scale.append(c_now)

            # Define transition block if not last layer
            if i < (num_blocks - 1):
                self.add_module('trans%d' % i_, DenseNetTransitionDown(
                    c_now, p_dropout=p_dropout,
                    compression_factor=compression_factor,
                    activation_fn=activation_fn,
                    normalization_fn=normalization_fn,
                ))
                c_now = list(self.children())[-1].c_now

            self.c_now = c_now

    def forward(self, x):
        # Apply initial layers and dense blocks
        for name, module in self.named_children():
            if name == 'initial':
                x, prev_scale_x = module(x)
            else:
                x = module(x)
        return x
