# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import FairseqIncrementalDecoder
from fairseq.models.transformer import TransformerConfig
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
)
from fairseq.modules import transformer_layer
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from torch import Tensor

from fairseq.modules.knn_datastore import KNN_Dstore

import pickle
from collections import Counter, OrderedDict
import os
import numpy as np

# rewrite name for backward compatibility in `make_generation_fast_`
def module_name_fordropout(module_name: str) -> str:
    if module_name == 'TransformerDecoderBase':
        return 'TransformerDecoder'
    else:
        return module_name


class TransformerDecoderBase(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *cfg.decoder.layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self,
        cfg,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.cfg = cfg
        super().__init__(dictionary)
        self.register_buffer("version", torch.Tensor([3]))
        self._future_mask = torch.empty(0)

        self.dropout_module = FairseqDropout(
            cfg.dropout, module_name=module_name_fordropout(self.__class__.__name__)
        )
        self.decoder_layerdrop = cfg.decoder.layerdrop
        self.share_input_output_embed = cfg.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = cfg.decoder.embed_dim
        self.embed_dim = embed_dim
        self.output_embed_dim = cfg.decoder.output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = cfg.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if cfg.no_scale_embedding else math.sqrt(embed_dim)

        if not cfg.adaptive_input and cfg.quant_noise.pq > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(embed_dim, embed_dim, bias=False),
                cfg.quant_noise.pq,
                cfg.quant_noise.pq_block_size,
            )
        else:
            self.quant_noise = None

        self.project_in_dim = (
            Linear(input_embed_dim, embed_dim, bias=False)
            if embed_dim != input_embed_dim
            else None
        )
        self.embed_positions = (
            PositionalEmbedding(
                self.max_target_positions,
                embed_dim,
                self.padding_idx,
                learned=cfg.decoder.learned_pos,
            )
            if not cfg.no_token_positional_embeddings
            else None
        )
        if cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layernorm_embedding = None

        self.cross_self_attention = cfg.cross_self_attention

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(cfg, no_encoder_attn)
                for _ in range(cfg.decoder.layers)
            ]
        )
        self.num_layers = len(self.layers)

        if cfg.decoder.normalize_before and not cfg.no_decoder_final_norm:
            self.layer_norm = LayerNorm(embed_dim, export=cfg.export)
        else:
            self.layer_norm = None

        self.project_out_dim = (
            Linear(embed_dim, self.output_embed_dim, bias=False)
            if embed_dim != self.output_embed_dim and not cfg.tie_adaptive_weights
            else None
        )

        self.adaptive_softmax = None
        self.output_projection = output_projection
        if self.output_projection is None:
            self.build_output_projection(cfg, dictionary, embed_tokens)

        self.fp16 = cfg.decoder.fp16

        self.knn_datastore = None
        if cfg.load_knn_datastore:
            self.knn_datastore = KNN_Dstore(cfg, len(dictionary))

        self.use_knn_datastore = cfg.use_knn_datastore
        self.knn_lambda_type = cfg.knn_lambda_type
        self.knn_lambda_threshold = cfg.knn_lambda_threshold
        

        self.knn_temperature_type = cfg.knn_temperature_type
        self.knn_search_prediction = cfg.knn_search_prediction
        self.use_knn_cache = cfg.knn_cache
        self.knn_search_every = cfg.knn_search_every
        self.knn_search_random = cfg.knn_search_random
        self.searching=True
        
        if self.use_knn_cache:
            self.knn_cache_threshold = cfg.knn_cache_threshold
            self.knn_cache=None
            self.knn_cache_probs=None

        self.use_faiss_centroids=cfg.use_faiss_centroids
        
        self.analyse=False

        if self.knn_lambda_threshold>0 or self.knn_search_prediction or self.use_knn_cache or self.use_faiss_centroids:
            self.need_to_search=0
            self.total_possible_searches=0

        self.chunk_size = cfg.chunk_size
        self.simplistic = cfg.simplistic
        self.unsupervised = cfg.unsupervised
        self.all_neighbours = cfg.all_neighbours
        self.knn_lambda_value_2 = cfg.knn_lambda_value_2
        self.retrieval_steps = cfg.retrieval_steps
        self.k = cfg.k

        self.supervised = cfg.supervised
        if self.supervised:
            self.knn_retrieval_lambda_linear = nn.Linear(embed_dim, 1, bias=False)
            self.knn_retrieval_sigmoid = nn.Sigmoid()
            self.training_supervised = cfg.training_supervised

        if self.retrieval_steps=='exp':
            self.retrieval_steps_max = cfg.retrieval_steps_max
            self.retrieval_steps_min = cfg.retrieval_steps_min

        elif self.retrieval_steps=='pred':
            self.retrieval_predictor = nn.Sequential(
                                nn.Linear(embed_dim+1, int(embed_dim*2)),
                                nn.ReLU(),
                                nn.Linear(int(embed_dim*2),1))
        
        self.intervals=[]
        self.mean_intervals=[]

        self.on_the_fly = cfg.on_the_fly
        self.on_the_fly_n_updates = cfg.on_the_fly_n_updates
        self.prev_example_idx = 0
        self.on_the_fly_reindexing = cfg.on_the_fly_reindexing
        self.on_the_fly_reindexing_interval = cfg.on_the_fly_reindexing_interval
        self.prev_reindexing_idx = 0

        self.on_the_fly_finetuning = cfg.on_the_fly_finetuning
        self.on_the_fly_finetuning_interval = cfg.on_the_fly_finetuning_interval

        self.knn_vary_c = cfg.knn_vary_c


    def build_output_projection(self, cfg, dictionary, embed_tokens):
        if cfg.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                utils.eval_str_list(cfg.adaptive_softmax_cutoff, type=int),
                dropout=cfg.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if cfg.tie_adaptive_weights else None,
                factor=cfg.adaptive_softmax_factor,
                tie_proj=cfg.tie_adaptive_proj,
            )
        elif self.share_input_output_embed:
            self.output_projection = nn.Linear(
                self.embed_tokens.weight.shape[1],
                self.embed_tokens.weight.shape[0],
                bias=False,
            )
            self.output_projection.weight = self.embed_tokens.weight
        else:
            self.output_projection = nn.Linear(
                self.output_embed_dim, len(dictionary), bias=False
            )
            nn.init.normal_(
                self.output_projection.weight, mean=0, std=self.output_embed_dim ** -0.5
            )
        num_base_layers = cfg.base_layers
        for i in range(num_base_layers):
            self.layers.insert(
                ((i + 1) * cfg.decoder.layers) // (num_base_layers + 1),
                BaseLayer(cfg),
            )

    def build_decoder_layer(self, cfg, no_encoder_attn=False):
        layer = transformer_layer.TransformerDecoderLayerBase(cfg, no_encoder_attn)
        checkpoint = cfg.checkpoint_activations
        if checkpoint:
            offload_to_cpu = cfg.offload_activations
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = cfg.min_params_to_wrap if not checkpoint else 0
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        features_only: bool = False,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
        src_lengths: Optional[Any] = None,
        return_all_hiddens: bool = False,
        new_sent: bool = False,
        newly_finished: Optional[Any] = None,
        beam_ids: Optional[Any] = None,
        example_idx: Optional[Any] = None,
        tgt_dict: Optional[Any] = None,
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention, should be of size T x B x C
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """

        """
        print('predicted tokens')
        for beam in prev_output_tokens:
            print([tgt_dict[token] for token in beam])
        
        print('\n')
        """

        x, extra = self.extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
        )

        if self.use_knn_datastore:
            last_hidden = x

        if not features_only and not self.supervised:
            x = self.output_layer(x)


        if self.use_knn_datastore:
            if self.on_the_fly and example_idx!=0 and example_idx%self.on_the_fly_n_updates <= 8 and example_idx-self.prev_example_idx > 8 and new_sent:
                self.knn_datastore.update_datastore(example_idx)
                self.prev_example_idx = example_idx

            if self.on_the_fly_reindexing and example_idx!=0 and example_idx%self.on_the_fly_reindexing_interval <= 8 and example_idx-self.prev_reindexing_idx > 8 and new_sent:
                self.knn_datastore.reindex_datastore(example_idx)
                self.prev_reindexing_idx = example_idx

            if self.use_knn_cache or self.knn_search_prediction or self.knn_lambda_threshold>0 or self.knn_search_every or self.knn_search_random or self.use_faiss_centroids:
                mask = torch.ones(last_hidden.size(0), dtype=torch.bool)
                knn_probs=torch.zeros(last_hidden.size(0), 1, self.output_projection.out_features).cuda()

            if self.supervised:
                if self.training_supervised:
                    for i in range(last_hidden.size(1)):
                        if i==0:
                            queries = last_hidden[:,i].unsqueeze(1)
                        elif i % self.chunk_size==0:
                            queries = torch.cat([queries, last_hidden[:,i].unsqueeze(1)],1)
                    
                    knn_search_result = self.knn_datastore.retrieve(queries)
                    tgt_index = knn_search_result['tgt_index']
                    tgt_index = tgt_index.unsqueeze(1).expand(-1,self.chunk_size,-1,-1)
                    tgt_index = tgt_index.contiguous().view(queries.size(0), -1,self.k,self.chunk_size)[:,:last_hidden.size(1)]

                    retrieved_tokens = self.embed_tokens(tgt_index).view(tgt_index.size(0),tgt_index.size(1),-1,last_hidden.size(-1))

                    retrieval_weights = torch.matmul(retrieved_tokens, last_hidden.unsqueeze(2).transpose(-1,-2))
                    retrieval_rep = torch.softmax(retrieval_weights,2).transpose(-1,-2).matmul(retrieved_tokens).squeeze()

                    retrieval_lambda = self.knn_retrieval_sigmoid(self.knn_retrieval_lambda_linear(last_hidden))
                    x = retrieval_lambda * retrieval_rep + (1-retrieval_lambda) * last_hidden

                    x = self.output_layer(x)
                    return x, extra

                else:
                    if new_sent:
                        self.knn_step=0
                    else:
                        self.knn_step+=1

                    if self.retrieval_steps=='chunk_size' and self.knn_step % self.chunk_size == 0:
                        knn_search_result = self.knn_datastore.retrieve(last_hidden)
                        tgt_index = knn_search_result['tgt_index']

                        self.retrieved_tokens = self.embed_tokens(tgt_index).view(tgt_index.size(0),-1,last_hidden.size(-1))
                    else:
                        self.retrieved_tokens = self.retrieved_tokens[[beam_ids.view(-1)]]

                    retrieval_weights = torch.matmul(self.retrieved_tokens, last_hidden.transpose(-1,-2))
                    retrieval_rep = torch.softmax(retrieval_weights,1).transpose(-1,-2).matmul(self.retrieved_tokens)
                    
                    retrieval_lambda = self.knn_retrieval_sigmoid(self.knn_retrieval_lambda_linear(last_hidden))

                    print(retrieval_lambda)

                    x = retrieval_lambda * retrieval_rep + (1-retrieval_lambda) * last_hidden
                    x = self.output_layer(x)

                    return x, extra


            elif self.chunk_size>1:
                if new_sent:
                    self.knn_step=0
                else:
                    self.knn_step+=1
                
                if self.retrieval_steps=='chunk_size':
                    if self.knn_step % self.chunk_size != 0:
                        if self.simplistic:
                            idx = self.knn_step % self.chunk_size

                            knn_prob = self.knn_datastore.compute_prob_simplistic(beam_ids, idx)

                            if self.knn_lambda_value_2==0:
                                knn_lambda = self.knn_datastore.get_lambda()
                            else:
                                knn_lambda = self.knn_lambda_value_2
                            
                            return x, extra, knn_prob, knn_lambda

                        elif self.unsupervised:
                            if self.knn_step % self.chunk_size==1:
                                chunk_start=True
                            else:
                                chunk_start=False

                            if self.knn_step==1:
                                sent_start=True
                            else:
                                sent_start=False

                            if self.all_neighbours:
                                neighbors = self.tgt_index[1].squeeze(1).squeeze(-1)
                                knn_index = self.knn_index.squeeze(1).squeeze(-1)
                            else:
                                neighbors = self.tgt_index[1][beam_ids.view(-1)].squeeze(1).squeeze(-1)
                                knn_index = self.knn_index[beam_ids.view(-1)].squeeze(1).squeeze(-1)
                            
                            knn_prob = self.knn_datastore.compute_prob_unsupervised(last_hidden, knn_index, neighbors, beam_ids, chunk_start=chunk_start, sent_start=sent_start, tgt_dict=tgt_dict)

                            if self.knn_lambda_value_2==0:
                                knn_lambda = self.knn_datastore.get_lambda()
                            else:
                                knn_lambda = self.knn_lambda_value_2

                            return x, extra, knn_prob, knn_lambda

                elif self.retrieval_steps=='exp':
                    if self.knn_step==0:
                        self.next_retrieval_step = self.knn_step + min(self.retrieval_steps_max, 
                                        int(self.retrieval_steps_min * 2 ** (1/2 * self.retrieval_steps_max * self.knn_step / encoder_out['encoder_out'][0].size(0))))
                        self.last_retrieval=0

                    elif self.knn_step!=self.next_retrieval_step:
                        if self.knn_step>=self.next_retrieval_step:
                            self.last_retrieval = self.next_retrieval_step
                            self.next_retrieval_step = self.knn_step + min(self.retrieval_steps_max, 
                                        int(self.retrieval_steps_min * 2 ** (1/2 * self.retrieval_steps_max * self.knn_step / encoder_out['encoder_out'][0].size(0))))
                        
                        if self.unsupervised:
                            if self.retrieval_steps == 'chunk_size':
                                if self.knn_step % self.chunk_size == 1:
                                    chunk_start=True
                                else:
                                    chunk_start=False
                            else:
                                if self.knn_step == self.last_retrieval+1:
                                    chunk_start=True
                                else:
                                    chunk_start=False

                            if self.knn_step == 1:
                                sent_start=True
                            else:
                                sent_start=False

                            if self.all_neighbours:
                                neighbors = self.tgt_index[1].squeeze(1).squeeze(-1)
                                knn_index = self.knn_index.squeeze(1).squeeze(-1)
                            else:
                                neighbors = self.tgt_index[1][beam_ids.view(-1)].squeeze(1).squeeze(-1)
                                knn_index = self.knn_index[beam_ids.view(-1)].squeeze(1).squeeze(-1)
                            

                            knn_prob = self.knn_datastore.compute_prob_unsupervised(last_hidden, knn_index, neighbors, beam_ids, chunk_start=chunk_start, sent_start=sent_start, tgt_dict=tgt_dict)

                            if knn_prob is not None:

                                if self.knn_lambda_value_2==0:
                                    knn_lambda = self.knn_datastore.get_lambda()
                                else:
                                    knn_lambda = self.knn_lambda_value_2

                                return x, extra, knn_prob, knn_lambda

                
                elif self.retrieval_steps=='pred':
                    print('Not implemented')

                else:
                    if self.unsupervised:
                        if self.knn_step==0:
                            self.retrieved=True
                            self.last_retrieval=0
                        else:
                            if self.knn_step==1:
                                sent_start=True
                            else:
                                sent_start=False

                            if self.all_neighbours:
                                neighbors = self.tgt_index[1].squeeze(1).squeeze(-1)
                                knn_index = self.knn_index.squeeze(1).squeeze(-1)
                            else:
                                neighbors = self.tgt_index[1][beam_ids.view(-1)].squeeze(1).squeeze(-1)
                                knn_index = self.knn_index[beam_ids.view(-1)].squeeze(1).squeeze(-1)
                            

                            knn_prob = self.knn_datastore.compute_prob_unsupervised(last_hidden, knn_index, neighbors, beam_ids, chunk_start=self.retrieved, sent_start=sent_start, tgt_dict=tgt_dict)

                            if knn_prob is not None:
                                self.retrieved=False

                                if self.knn_lambda_value_2==0:
                                    knn_lambda = self.knn_datastore.get_lambda()
                                else:
                                    knn_lambda = self.knn_lambda_value_2

                                return x, extra, knn_prob, knn_lambda
                            else:
                                self.retrieved=True
                                
                                if self.last_retrieval==0 and len(self.intervals)>1:
                                    self.mean_intervals.append(np.array(self.intervals).mean())
                                    self.intervals=[]
                                
                                self.intervals.append(self.knn_step-self.last_retrieval)

                                self.last_retrieval = self.knn_step
                                if len(self.mean_intervals)>0:
                                    print(np.array(self.mean_intervals).mean())
                                

                
            if self.knn_search_every>0:
                if new_sent:
                    self.knn_step=0
                if self.knn_step % self.knn_search_every!=0:
                    mask[:] = False
                    if not self.use_knn_cache:
                        last_hidden=last_hidden[mask]
                    self.knn_step+=1
                    self.searching=False
                else:
                    self.knn_step+=1
                    self.searching=True

            elif self.knn_search_random>0:
                aux = torch.rand(1)
                if aux<self.knn_search_random:
                    self.searching=True
                else:
                    self.searching=False
                    mask[:] = False
                    if not self.use_knn_cache:
                        last_hidden=last_hidden[mask]

            if self.use_knn_cache:
                if new_sent:
                    self.knn_cache=None
                    self.knn_cache_probs=None
        
                if self.knn_cache is not None:
                    dists = torch.cdist(last_hidden.squeeze(1), self.knn_cache.squeeze(1), p=2).min(-1)

                    self.knn_cache = torch.cat([self.knn_cache, last_hidden],0)

                    indices = (dists.values<=self.knn_cache_threshold).nonzero()[:,0]
                    mask[indices] = False
                    if not self.use_faiss_centroids:
                        last_hidden=last_hidden[mask]


                    if indices.size(0)>0:
                        knn_probs[indices] = self.knn_cache_probs[dists.indices[indices]]

                    self.need_to_search += x.size(0) - indices.size(0)
                    self.total_possible_searches+=x.size(0)

                else:
                    self.knn_cache=last_hidden

      
            knn_lambda = self.knn_datastore.get_lambda()

            if ((self.knn_lambda_threshold == 0 and not self.knn_search_prediction and not self.use_knn_cache and not self.use_faiss_centroids) or last_hidden.size(0) > 0) and self.searching:
                if self.knn_vary_c:
                    knn_search_result = self.knn_datastore.retrieve(last_hidden, c=int(self.next_retrieval_step-self.last_retrieval))
                else:
                    knn_search_result = self.knn_datastore.retrieve(last_hidden)

                knn_dists = knn_search_result['distance']  # [batch, seq len, k]  # we need do sort
                knn_index = knn_search_result['knn_index']
                tgt_index = knn_search_result['tgt_index']

                knn_temperature = self.knn_datastore.get_temperature()

                decode_result = self.knn_datastore.calculate_knn_prob(knn_index, tgt_index, knn_dists, last_hidden, knn_temperature)
                knn_prob = decode_result['prob']

                if self.chunk_size > 1 and self.simplistic:
                    knn_prob = decode_result['prob']
                    self.tgt_index = tgt_index

                elif self.chunk_size > 1 and self.unsupervised:
                    knn_prob = decode_result['prob']
                    self.tgt_index = tgt_index
                    self.knn_index = knn_index

                if self.knn_lambda_threshold > 0 or self.knn_search_prediction or self.use_knn_cache or self.use_faiss_centroids:    
                    knn_probs[mask]=knn_prob
                    if self.use_knn_cache:
                        if self.knn_cache_probs is None:
                            self.knn_cache_probs=knn_probs
                        else:
                            self.knn_cache_probs=torch.cat([self.knn_cache_probs, knn_probs],0)

                    return x, extra, knn_probs, knn_lambda, knn_dists, knn_index

            elif not self.use_knn_cache:
                knn_dists = 0
                knn_index = 0
                tgt_index = 0
                
                knn_probs=torch.zeros(x.size(0), 1, self.output_projection.out_features).cuda()
            
                return x, extra, knn_probs, knn_lambda, knn_dists, knn_index

            else:
                if self.knn_cache_probs is None:
                    self.knn_cache_probs=knn_probs
                else:
                    self.knn_cache_probs=torch.cat([self.knn_cache_probs, knn_probs],0)

                knn_dists = 0
                knn_index = 0
                tgt_index = 0
                return x, extra, knn_probs, knn_lambda, knn_dists, knn_index

            if features_only:
                prob = utils.softmax(self.output_layer(x), dim=-1, onnx_trace=self.onnx_trace)
                return x, extra, knn_prob, prob, prev_output_tokens

            if self.analyse:
                return x, extra, knn_prob, knn_lambda, knn_dists, knn_index, last_hidden

            return x, extra, knn_prob, knn_lambda, knn_dists, knn_index

        else:
            return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        return self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )

    """
    A scriptable subclass of this class has an extract_features method and calls
    super().extract_features, but super() is not supported in torchscript. A copy of
    this function is made to be used in the subclass instead.
    """

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        if encoder_out is not None and len(encoder_out["encoder_out"]) > 0:
            enc = encoder_out["encoder_out"][0]
            assert (
                enc.size()[1] == bs
            ), f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
        if encoder_out is not None and len(encoder_out["encoder_padding_mask"]) > 0:
            padding_mask = encoder_out["encoder_padding_mask"][0]

        # embed positions
        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def output_layer(self, features):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            return self.output_projection(features)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions)

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                utils.fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = "{}.embed_positions.weights".format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict[
                "{}.embed_positions._float_tensor".format(name)
            ] = torch.FloatTensor(1)

        if f"{name}.output_projection.weight" not in state_dict:
            if self.share_input_output_embed:
                embed_out_key = f"{name}.embed_tokens.weight"
            else:
                embed_out_key = f"{name}.embed_out"
            if embed_out_key in state_dict:
                state_dict[f"{name}.output_projection.weight"] = state_dict[embed_out_key]
                if not self.share_input_output_embed:
                    del state_dict[embed_out_key]

        for i in range(self.num_layers):
            # update layer norms
            layer_norm_map = {
                "0": "self_attn_layer_norm",
                "1": "encoder_attn_layer_norm",
                "2": "final_layer_norm",
            }
            for old, new in layer_norm_map.items():
                for m in ("weight", "bias"):
                    k = "{}.layers.{}.layer_norms.{}.{}".format(name, i, old, m)
                    if k in state_dict:
                        state_dict[
                            "{}.layers.{}.{}.{}".format(name, i, new, m)
                        ] = state_dict[k]
                        del state_dict[k]

        version_key = "{}.version".format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        #for n in state_dict.keys():
        #    print(n)
        #    print(state_dict[n])

        return state_dict


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m


class TransformerDecoder(TransformerDecoderBase):
    def __init__(
        self,
        args,
        dictionary,
        embed_tokens,
        no_encoder_attn=False,
        output_projection=None,
    ):
        self.args = args
        super().__init__(
            TransformerConfig.from_namespace(args),
            dictionary,
            embed_tokens,
            no_encoder_attn=no_encoder_attn,
            output_projection=output_projection,
        )

    def build_output_projection(self, args, dictionary, embed_tokens):
        super().build_output_projection(
            TransformerConfig.from_namespace(args), dictionary, embed_tokens
        )

    def build_decoder_layer(self, args, no_encoder_attn=False):
        return super().build_decoder_layer(
            TransformerConfig.from_namespace(args), no_encoder_attn=no_encoder_attn
        )
