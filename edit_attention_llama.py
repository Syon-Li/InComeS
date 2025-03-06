#  transformer v4.45.2

import torch
import random
import warnings
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union, Dict
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaSdpaAttention,
    LlamaDecoderLayer,
    LlamaModel,
    LlamaForCausalLM,
    apply_rotary_pos_emb,
    _prepare_4d_causal_attention_mask_with_cache_position,
    repeat_kv,
    AttentionMaskConverter,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging, is_torchdynamo_compiling
# from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
from utils import reverse_cumsum, alter_position_ids, sattolo_cycle
import os


logger = logging.get_logger(__name__)



class Edit_LlamaSdpaAttention(LlamaSdpaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.zero_gist_key = nn.parameter.Parameter(data=torch.randn(1, self.num_key_value_heads, self.head_dim))
        self.n_layers = config.num_hidden_layers
        if layer_idx < self.n_layers//2:
            self.zero_gist_key.requires_grad = False
            

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            gist_pool: Optional[Dict[int, Dict]] = None, #[num_gist+1, num_key_value_head, head_dim]
            gist_idx_vector: Optional[torch.Tensor] = None, #[bsz, q_len]
            gist_pool_idx: Optional[torch.Tensor] = None, #[bsz, num_gist+1]
            topk: Optional[int] = None,   
            sel_indices: Optional[List] = None, 
            extra_loss: Optional[Dict[str, List]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

            if output_attentions:
                # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
                logger.warning_once(
                    "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                    'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
                return super().forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)         

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


            for _,v in gist_pool.items():
                v['keys'] = v['keys'].to(query_states.device)
                v["values"] = v["values"].to(query_states.device)

            # updating new gist
            if self.layer_idx >= self.n_layers//2 and gist_idx_vector.any().item():
                new_gist_keys = key_states.transpose(1,2)[gist_idx_vector]
                new_gist_values = value_states.transpose(1,2)[gist_idx_vector]                               
                gist_pool[self.layer_idx]["keys"] = torch.concat([gist_pool[self.layer_idx]["keys"], new_gist_keys]).to(query_states.device)
                gist_pool[self.layer_idx]["values"] = torch.concat([gist_pool[self.layer_idx]["values"], new_gist_values]).to(query_states.device)


            raw_gist_keys = gist_pool[self.layer_idx]["keys"]
            raw_gist_values = gist_pool[self.layer_idx]["values"]   


            # if self.training and gist_pool_idx is not None and num_gist>=4:
            #     for _ in range(3):
            #         permu = sattolo_cycle(num_gist)
            #         na = random.uniform(0.2, 0.8)
            #         # print("permu", permu)
            #         gist_keys_aug = na*gist_pool[self.layer_idx]["keys"][permu] + (1-na)*gist_pool[self.layer_idx]["keys"]
            #         gist_values_aug = na*gist_pool[self.layer_idx]["values"][permu] + (1-na)*gist_pool[self.layer_idx]["values"]

            #         # aug_n = 68
            #         # aug_w = torch.randn(aug_n, num_gist, device=query_states.device)
            #         # aug_w[...,0] = aug_w[...,0] + torch.finfo(query_states.dtype).min
            #         # aug_w = aug_w.softmax(dim=-1)
            #         # gist_keys_aug = aug_w.matmul(raw_gist_keys.reshape(num_gist, -1))
            #         # gist_values_aug = aug_w.matmul(raw_gist_values.reshape(num_gist, -1))
            #         # gist_keys_aug = gist_keys_aug.reshape(aug_n, -1, self.head_dim)
            #         # gist_values_aug = gist_values_aug.reshape(aug_n, -1, self.head_dim)

            #         raw_gist_keys = torch.concat([raw_gist_keys, gist_keys_aug]).to(query_states.dtype)
            #         raw_gist_values = torch.concat([raw_gist_values, gist_values_aug]).to(query_states.dtype)
            #         gist_pool_idx = torch.concat([gist_pool_idx, torch.zeros(bsz, gist_keys_aug.shape[0], device=query_states.device)], dim=-1)

                # print("raw_gist_keys", raw_gist_keys.shape)
                # print("raw_gist_values", raw_gist_values.shape)


            # #Add zero gist key and value
            zero_values = torch.zeros(1, self.num_key_value_heads, self.head_dim, device=query_states.device)
            gist_keys = torch.concat([self.zero_gist_key, raw_gist_keys]).to(query_states.dtype) # [num_gist+1,key_value_head,head_dim]
            gist_values = torch.concat([zero_values, raw_gist_values]).to(query_states.dtype)   

            # num_gist = gist_keys.shape[0]
            # gist_pos_ids = -torch.ones(num_gist, device=gist_keys.device)
            # # print("gist_pos_ids", gist_pos_ids)
            # g_cos, g_sin = self.rotary_emb(gist_keys, gist_pos_ids[None,...])
            # _, gist_keys = apply_rotary_pos_emb(gist_keys[None,...], gist_keys[None,...], cos=g_cos, sin=g_sin, unsqueeze_dim=2)
            # gist_keys = gist_keys[0,...]  


            if position_embeddings is None:
                logger.warning_once(
                    "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                    "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                    "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                    "removed and `position_embeddings` will be mandatory."
                )
                cos, sin = self.rotary_emb(value_states, position_ids)
            else:
                cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


            #updating new gist
            # if gist_idx_vector.any().item():
            #     new_gist_keys = key_states.transpose(1,2)[gist_idx_vector]
            #     new_gist_values = value_states.transpose(1,2)[gist_idx_vector]                               
            #     gist_pool[self.layer_idx]["keys"] = torch.concat([gist_pool[self.layer_idx]["keys"].to(query_states.device), new_gist_keys])
            #     gist_pool[self.layer_idx]["values"] = torch.concat([gist_pool[self.layer_idx]["values"].to(query_states.device), new_gist_values])                     

            # # Add zero gist key and value
            # zero_values = torch.zeros(1, self.num_key_value_heads, self.head_dim, device=query_states.device)
            # gist_keys = torch.concat([self.zero_gist_key, gist_pool[self.layer_idx]["keys"]]).to(query_states.dtype) # [num_gist+1,key_value_head,head_dim]
            # gist_values = torch.concat([zero_values, gist_pool[self.layer_idx]["values"]]).to(query_states.dtype)   

            # num_gist = gist_keys.shape[0]



            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            causal_mask, atten_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]


            # gist_keys = gist_keys.repeat_interleave(self.num_key_value_groups, dim=1).transpose(0,1)
            # gist_values = gist_values.repeat_interleave(self.num_key_value_groups, dim=1).transpose(0,1)    
            # gist_logits = torch.matmul(query_states.to(gist_keys.dtype), gist_keys.transpose(1, 2)) / math.sqrt(self.head_dim) # [bsz, num_heads, q_len, num_gist]
            num_gist = gist_keys.shape[0]
            gist_keys = gist_keys[:,:,None,:].expand(-1, -1, self.num_key_value_groups, -1)
            gist_values = gist_values[:,:,None,:].expand(-1, -1, self.num_key_value_groups, -1)
            gist_keys = gist_keys.reshape(self.num_heads, self.head_dim, num_gist)
            gist_values = gist_values.reshape(self.num_heads, num_gist, self.head_dim)
            gist_logits = torch.matmul(query_states.to(gist_keys.dtype), gist_keys) / math.sqrt(self.head_dim) # [bsz, num_heads, q_len, num_gist]


            # print("topk", topk)
            # topk=32
            # if gist_logits.shape[-1] > topk:
            #     sorted, indices = gist_logits.sort(dim=-1, descending=True)
            #     sorted_topk = sorted[...,topk-1]
            #     # gist_logits_mask = (gist_logits < sorted_topk[...,None]).to(query_states.dtype)

            #     # Make sure the right gist and zero gist are not masked
            #     # if self.training and gist_pool_idx is not None:
            #     #     gist_pool_idx[...,0] = 1
            #     #     # print("gist_pool_idx", gist_pool_idx)
            #     #     gist_logits_mask = gist_logits_mask * gist_pool_idx[:,None,None,:].logical_not().to(query_states.dtype)

            #     # gist_logits_mask = gist_logits_mask * torch.finfo(query_states.dtype).min
            #     # gist_logits = gist_logits + gist_logits_mask
                
            #     if self.training:
            #         if gist_pool_idx is not None:
            #             gist_logits_mask = (gist_logits >= sorted_topk[...,None])
            #             # gist_logits_mask[...,0] = True
            #             # gist_logits_mask = gist_logits_mask.bitwise_or(gist_pool_idx[:,None,None,:].to(torch.bool))
            #             gist_logits_mask = gist_logits_mask.to(query_states.dtype) * self.offset
            #             gist_logits = gist_logits + gist_logits_mask
            #     else:
            #         gist_logits_mask = (gist_logits < sorted_topk[...,None]).to(query_states.dtype)
            #         # gist_logits_mask[...,0] = 0
            #         gist_logits_mask = gist_logits_mask * torch.finfo(query_states.dtype).min
            #         gist_logits = gist_logits + gist_logits_mask

                    # gist_logits_mask = (gist_logits >= sorted_topk[...,None]).to(query_states.dtype)
                    # gist_logits_mask = gist_logits_mask * 2
                    # gist_logits = gist_logits + gist_logits_mask
            


            # sorted, indices = gist_logits.sort(dim=-1, descending=True)
            # rank_base = torch.arange(gist_logits.shape[-1], device=gist_logits.device, dtype=gist_logits.dtype) / (gist_logits.shape[-1] - 1)
            # rank_base = torch.zeros_like(gist_logits).scatter_(dim=-1, index=indices, src=rank_base.expand(indices.shape))
            # # print(rank_expo.expand(indices.shape))
            # if self.training and gist_pool_idx is not None:
            #     gist_logits = gist_logits - (-rank_base**(1e-4*gist_logits.shape[-1]) + 1) * math.log10(gist_logits.shape[-1]) * gist_pool_idx[:,None,None,:].to(gist_logits.dtype)
            #     # print((-rank_base**(1e-4*gist_logits.shape[-1]) + 1) * math.log10(gist_logits.shape[-1]) * gist_pool_idx[:,None,None,:].to(gist_logits.dtype))
            
            # if not self.training:
            #     gist_logits = gist_logits + (-rank_base**(1e-4*100) + 1) * math.log10(gist_logits.shape[-1])  


            # topk=8
            # sorted, indices = gist_logits.sort(dim=-1, descending=True)
            # sorted_topk = sorted[...,topk]
            # if self.layer_idx > 0 and self.layer_idx < 7:
            #     sel_indices = sel_indices.bitwise_or(gist_logits > sorted_topk[...,None])
            #     # gist_logits_mask = sel_indices.logical_not().to(gist_logits.dtype) * torch.finfo(query_states.dtype).min
            #     # gist_logits = gist_logits + gist_logits_mask
            # # if self.layer_idx == 6:
            # #     sel_indices = sel_indices.bitwise_or(gist_logits > sorted_topk[...,None])
            # elif self.layer_idx == 0:
            #     sel_indices = torch.zeros(gist_logits.shape, dtype=torch.bool, device=gist_logits.device)
            #     # sel_indices = torch.ones(gist_logits.shape, dtype=torch.bool, device=gist_logits.device)
            # elif self.layer_idx >= 7:
            #     # if self.layer_idx == 15:
            #     #     indices_sort, idx = sel_indices.sum(dim=1).sort(dim=-1, descending=True)
            #     #     print(indices_sort[...,:8*2])
            #     #     print(idx[...,:8*2])
            #     gist_logits_mask = sel_indices.logical_not().to(gist_logits.dtype) * torch.finfo(query_states.dtype).min
            #     gist_logits = gist_logits + gist_logits_mask


            # if self.layer_idx >= 7:
            #     gist_logits_mask = (gist_logits < sorted_topk[...,None]).to(query_states.dtype)
            #     gist_logits_mask = gist_logits_mask * torch.finfo(query_states.dtype).min
            #     gist_logits = gist_logits + gist_logits_mask





            # Add a residual to the zero gist
            # gist_logits = gist_logits.clone()
            # gist_logits[...,0] = gist_logits[...,0] + 7
            # gist_logits[...,0] = gist_logits[...,0] - 5
            # # gist_logits[...,0] = gist_logits[...,0]*2

            # Mask the gist token selection
            # if self.training and gist_pool_idx is not None:
            #     gist_pool_idx[...,0] = 1
            #     gist_logits_mask = torch.zeros(gist_logits.shape, device=gist_logits.device)
            #     masked_fill = gist_pool_idx[:,None,:].repeat_interleave(repeats=q_len, dim=1).logical_not()
            #     gist_logits_mask = gist_logits_mask.masked_fill(masked_fill[:,None,...], value=torch.finfo(hidden_states.dtype).min)
            #     # print("gist_logits_mask", gist_logits_mask, gist_logits_mask.shape)
            #     gist_logits = gist_logits + gist_logits_mask





            # print("dropout", self.attention_dropout)
            # if self.layer_idx > 7:
            #     gist_logits = gist_logits / 0.45
            # else:
            #     gist_logits = gist_logits / 0.9
            gist_weights = nn.functional.softmax(gist_logits, dim=-1, dtype=torch.float32).to(query_states.dtype) # [bsz, num_heads, q_len, gist_num]
            # if self.layer_idx > 0 and self.layer_idx < 7:
            # # if self.layer_idx >= 7:
            #     prob_sort, idx = gist_weights.sort(dim=-1, descending=True)
            #     print(self.layer_idx)
            #     print(prob_sort[...,:8])
            #     print(idx[...,:8])
                
            gist_weights = nn.functional.dropout(gist_weights, p=self.attention_dropout, training=self.training)
            atten_mask = atten_mask.to(key_states.dtype).clone()
            atten_mask[reverse_cumsum(gist_idx_vector)>0] = 0
            gist_weights = gist_weights * atten_mask[:,None,:,None]

            gist_output = torch.matmul(gist_weights, gist_values) # [bsz, num_heads, q_len, head_dim]
            


            if not self.training:
                eps = 1e-5
                # Entropy
                sparsity_loss = -gist_weights * (gist_weights+eps).log()
                sparsity_loss = ((sparsity_loss.sum(dim=-1) * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
                # print("sparsity_loss", sparsity_loss)
                if extra_loss is not None:
                    # extra_loss["Entropy"].append(sparsity_loss)        
                    extra_loss["Entropy"].append((self.layer_idx, sparsity_loss))   
        

        

            if self.training and gist_pool_idx is not None and self.layer_idx >= self.n_layers//2:
                eps = 1e-5

                p0_loss = -(gist_weights[...,0] + eps).log()
                p0_loss = ((p0_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()

                if torch.isnan(p0_loss).item():
                    print("gist_weight with NAN", gist_weights, gist_weights.sum(-1), gist_weights.shape)
                    print("attention_mask", atten_mask, atten_mask.sum(-1))
                
                # print("p0_loss", p0_loss)
                if p0_loss!=0:
                    extra_loss["p0_loss"].append(p0_loss)


                gist_pool_idx[...,0] = 0
                mask = atten_mask[...,None].matmul(gist_pool_idx[:,None,:].to(atten_mask.dtype))
                # print("mask", mask, mask.shape)

                pS_loss = -((gist_weights * mask[:,None,:,:]).sum(-1) + eps).log()
                pS_loss = ((pS_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()

                # print("pS_loss", pS_loss)
                # if self.layer_idx>=4 and self.layer_idx<=11:
                if pS_loss!=0:
                    extra_loss["pS_loss"].append(pS_loss)


                # Entropy
                sparsity_loss = -gist_weights * (gist_weights+eps).log()
                sparsity_loss = ((sparsity_loss.sum(dim=-1) * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
                # print("sparsity_loss", sparsity_loss)
                if sparsity_loss!=0:
                    extra_loss["sparsity_loss"].append(sparsity_loss)
            
                # print(p0_loss, pS_loss, sparsity_loss)



            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            is_causal = True if causal_mask is None and q_len > 1 else False

            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask.to(query_states.dtype),
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            # if self.layer_idx >= 7:
            attn_output = attn_output + gist_output
            # print("after_attn_output", attn_output, attn_output.shape)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)

            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value, sel_indices
    






class Edit_LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.self_attn = Edit_LlamaSdpaAttention(config=config, layer_idx=layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_idx_vector: Optional[torch.Tensor] = None,    
        gist_pool_idx: Optional[torch.Tensor] = None,    
        topk: Optional[int] = None,   
        sel_indices: Optional[List] = None, 
        extra_loss: Optional[Dict[str, List]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, sel_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            gist_idx_vector=gist_idx_vector,
            gist_pool=gist_pool,
            gist_pool_idx=gist_pool_idx,
            topk=topk,
            sel_indices=sel_indices, 
            extra_loss=extra_loss,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs, sel_indices
   



class Edit_LlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Edit_LlamaDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])


    # make sure the causal mask is always returned, since we will further make modification on it
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa":
            # if AttentionMaskConverter._ignore_causal_mask_sdpa(
            #     attention_mask,
            #     inputs_embeds=input_tensor,
            #     past_key_values_length=past_seen_tokens,
            #     is_training=self.training,
            # ):
            #     return None
            pass


        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )
        
        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,     
        gist_pool_idx: Optional[torch.Tensor] = None,
        topk: Optional[int] = 8,    
        sel_indices: Optional[List] = None,       
        extra_loss: Optional[Dict[str, List]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # print("input_embeds", inputs_embeds.dtype)

        
        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        #gist location boolean vector
        gist_idx_vector = (input_ids == gist_token_ids)

        # print("position_ids", position_ids, position_ids.shape)
        position_ids = alter_position_ids(gist_token_ids=gist_token_ids, input_ids=input_ids, origin_pos_ids=position_ids)
        # print(position_ids)

        # print("self.config._attn_implementation", self.config._attn_implementation)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions) # [bsz, 1, q_len, q_len]
        # print("causal_mask:", causal_mask, causal_mask.shape)

        
        # if gist_idx_vector.any().item():
        col_segment = (reverse_cumsum(gist_idx_vector) > 0).to(inputs_embeds.dtype)
        # row_segment = ((gist_idx_vector.cumsum(-1).cumsum(-1) - 1) > 0).to(inputs_embeds.dtype)
        row_segment = col_segment.logical_not().to(inputs_embeds.dtype)
        mask1 = row_segment[...,None].matmul(col_segment[:,None,:])
        attention_mask = attention_mask.to(inputs_embeds.dtype)
        mask1[attention_mask[...,None].matmul(attention_mask[:,None,:]).logical_not()] = 0  # avoid changing the pad token value in causal mask 
        # print("gist_idx_vector", gist_idx_vector, gist_idx_vector.sum())
        # print("mask1", mask1, mask1.sum(), mask1.shape)
        causal_mask[mask1[:,None,...].to(torch.bool)] = torch.finfo(inputs_embeds.dtype).min
        # print("modified casual_mask", causal_mask, causal_mask.shape)

    

        # embed positions
        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        sel_indices = []
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs, sel_indices = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    (causal_mask, attention_mask),
                    gist_idx_vector,
                    gist_pool,
                    gist_pool_idx,
                    topk,
                    sel_indices, 
                    extra_loss,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs, sel_indices = decoder_layer(
                    hidden_states,
                    attention_mask=(causal_mask, attention_mask),
                    gist_idx_vector=gist_idx_vector,
                    gist_pool=gist_pool,
                    gist_pool_idx=gist_pool_idx,
                    topk=topk,
                    sel_indices=sel_indices, 
                    extra_loss=extra_loss,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]
            

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )





class Edit_LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig,):
        super().__init__(config)
        self.model = Edit_LlamaModel(config)
        self.gist_pool = None
        self.gist_token_ids = config.vocab_size + 1
        self.gist_pool_idx = None
        self.topk = 8



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,
        gist_pool_idx: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
        extra_loss: Optional[Dict[str, List]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            gist_pool=gist_pool if gist_pool is not None else self.gist_pool,
            gist_token_ids=gist_token_ids if gist_token_ids is not None else self.gist_token_ids,
            gist_pool_idx=gist_pool_idx if gist_pool_idx is not None else self.gist_pool_idx,
            topk=topk if topk is not None else self.topk,
            extra_loss=extra_loss,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            if labels is None and not is_torchdynamo_compiling():
                logger.warning_once(
                    "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
                )
            # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
            # TODO: remove the float() operation in v4.46
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            # loss_fct = LigerFusedLinearCrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            # print("loss", loss)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    