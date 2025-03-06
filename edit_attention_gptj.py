#  transformer v4.45.2

import torch
import random
import warnings
import itertools
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union, Dict
from transformers.models.gptj.modeling_gptj import (
    GPTJAttention,
    GPTJBlock,
    GPTJModel,
    GPTJForCausalLM,
    get_embed_positions,
    is_torch_fx_proxy,
    apply_rotary_pos_emb,
    logger,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from utils import reverse_cumsum, alter_position_ids
from liger_kernel.transformers import liger_rotary_pos_emb, LigerLayerNorm




# Customized RoPE that supports negative pos id value, reference to LLama RoPE
@torch.no_grad()
def create_sinusoidal(x, position_ids, dim:int):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim)).to(x.device)

    # Core RoPE block
    inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # emb = torch.cat((freqs, freqs), dim=-1)
        emb = freqs
        cos = emb.cos()
        sin = emb.sin()

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)



# modified GPT-J attention using sdpa and liger kernel RoPE, and layernorm
class Edit_GPTJAttention(GPTJAttention):
    def __init__(self, config, layer_idx=None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.zero_gist_key = nn.parameter.Parameter(data=torch.randn((1, config.num_attention_heads, config.hidden_size // config.num_attention_heads)))
        self.n_layers = config.n_layer
        if layer_idx < self.n_layers//2:
            self.zero_gist_key.requires_grad = False


    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        gist_pool: Optional[Dict[int, Dict]] = None, #[num_gist+1, num_key_value_head, head_dim]
        gist_idx_vector: Optional[torch.Tensor] = None, #[bsz, q_len]
        gist_pool_idx: Optional[torch.Tensor] = None, #[bsz, num_gist+1]
        topk: Optional[int] = None,
        extra_loss: Optional[Dict[str, List]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        
        bsz, q_len, _ = hidden_states.size()

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.num_attention_heads, self.head_dim, True) # (bsz, q_len, heads, head_dim)
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, True) # (bsz, q_len, heads, head_dim)
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, False) # (bsz, heads, q_len, head_dim)
    

        for _,v in gist_pool.items():
            v['keys'] = v['keys'].to(query.device)
            v["values"] = v["values"].to(query.device)

        # updating new gist
        if self.layer_idx >= self.n_layers//2 and gist_idx_vector.any().item():
            new_gist_keys = key[gist_idx_vector]
            new_gist_values = value.transpose(1,2)[gist_idx_vector]                               
            gist_pool[self.layer_idx]["keys"] = torch.concat([gist_pool[self.layer_idx]["keys"].to(query.device), new_gist_keys])
            gist_pool[self.layer_idx]["values"] = torch.concat([gist_pool[self.layer_idx]["values"].to(query.device), new_gist_values])   

        #Add zero gist key and value
        zero_values = torch.zeros((1, self.num_attention_heads, self.head_dim)).to(query.device)
        gist_keys = torch.concat([self.zero_gist_key, gist_pool[self.layer_idx]["keys"]]).to(query.dtype) # [gist_num+1,key_value_head,head_dim]
        gist_values = torch.concat([zero_values, gist_pool[self.layer_idx]["values"]]).to(query.dtype)             

        # num_gist = gist_keys.shape[0]      


        # position_ids[position_ids<0] = position_ids[position_ids<0] + self.config.max_position_embeddings
        # print(position_ids, position_ids.shape)
        # if is_torch_fx_proxy(position_ids) or torch.jit.is_tracing():
        #     # The logic to conditionally copy to GPU could not be traced, so we do this
        #     # every time in the torch.fx case
        #     embed_positions = get_embed_positions(self.embed_positions, position_ids)
        # else:
        #     embed_positions = self._get_embed_positions(position_ids)

        # repeated_position_ids = position_ids.unsqueeze(-1).repeat(1, 1, embed_positions.shape[-1])
        # # print(embed_positions, embed_positions.shape)
        # print(repeated_position_ids, repeated_position_ids.shape)
        # sincos = torch.gather(embed_positions, 1, repeated_position_ids)
        # sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        cos, sin = create_sinusoidal(x=query, position_ids=position_ids, dim=self.rotary_dim or self.embed_dim)

        # gist_pos_ids = -torch.ones(num_gist, device=gist_keys.device)
        # # print("gist_pos_ids", gist_pos_ids)
        # g_cos, g_sin = create_sinusoidal(x=gist_keys, position_ids=gist_pos_ids[None,...], dim=self.rotary_dim or self.embed_dim)


        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            # gist_keys_rot = gist_keys[...,:self.rotary_dim]
            # gist_keys_pass = gist_keys[...,self.rotary_dim:]

            # k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            # q_rot = apply_rotary_pos_emb(q_rot, sin, cos)
            q_rot, k_rot = liger_rotary_pos_emb(q=q_rot.transpose(1,2), k=k_rot.transpose(1,2), cos=cos, sin=sin)

            # gist_keys_rot = apply_rotary_pos_emb(gist_keys_rot[None,...], g_sin, g_cos)
            # _, gist_keys_rot = liger_rotary_pos_emb(gist_keys_rot[None,...].transpose(1,2), gist_keys_rot[None,...].transpose(1,2), sin=g_sin, cos=g_cos)

            key = torch.cat([k_rot.transpose(1,2), k_pass], dim=-1)
            query = torch.cat([q_rot.transpose(1,2), q_pass], dim=-1)
            # gist_keys = torch.cat([gist_keys_rot[0,...].transpose(0,1), gist_keys_pass], dim=-1)
        else:
            # key = apply_rotary_pos_emb(key, sin, cos)
            # query = apply_rotary_pos_emb(query, sin, cos)
            # gist_keys = apply_rotary_pos_emb(gist_keys[None,...], g_sin, g_cos)
            # gist_keys = gist_keys[0,...]
            query, key = liger_rotary_pos_emb(q=query.transpose(1,2), k=key.transpose(1,2), cos=cos, sin=sin)
            # _, gist_keys = liger_rotary_pos_emb(q=gist_keys[None,...].transpose(1,2), k=gist_keys[None,...].transpose(1,2), sin=g_sin, cos=g_cos)
            # gist_keys = gist_keys[0,...].transpose(0,1)

        key = key.permute(0, 2, 1, 3) # (bsz, head, q_len, head_features)
        query = query.permute(0, 2, 1, 3) 


        causal_mask, atten_mask = attention_mask
        atten_mask = atten_mask.to(key.dtype).clone()

        gist_logits = torch.matmul(query, gist_keys.permute(1,2,0)) / math.sqrt(self.head_dim) # [bsz, num_heads, q_len, gist_num+1]
        # print(gist_logits.shape, gist_logits1.shape)

        # if gist_logits.shape[-1] > topk:
        #     sorted, indices = gist_logits.sort(dim=-1, descending=True)
        #     sorted_topk = sorted[...,topk-1]
        #     # gist_logits_mask = (gist_logits < sorted_topk[...,None]).to(torch.float)

        #     # Make sure the right gist and zero gist are not masked
        #     # if self.training and gist_pool_idx is not None:
        #     #     gist_pool_idx[...,0] = 1
        #     #     # print("gist_pool_idx", gist_pool_idx)
        #     #     gist_logits_mask = gist_logits_mask * gist_pool_idx[:,None,None,:].logical_not().to(query.dtype)

        #     # gist_logits_mask = gist_logits_mask * torch.finfo(query.dtype).min
        #     # gist_logits = gist_logits + gist_logits_mask

        #     if self.training:
        #         gist_logits_mask = (gist_logits >= sorted_topk[...,None]).to(query.dtype) * 10
        #         gist_logits = gist_logits + gist_logits_mask
        #     else:
        #         gist_logits_mask = (gist_logits < sorted_topk[...,None]).to(query.dtype)
        #         gist_logits_mask = gist_logits_mask * torch.finfo(query.dtype).min
        #         gist_logits = gist_logits + gist_logits_mask


        

        

        gist_weights = nn.functional.softmax(gist_logits, dim=-1, dtype=torch.float32).to(query.dtype) # [bsz, num_heads, q_len, gist_num+1]
        gist_weights = self.attn_dropout(gist_weights)
        

        # print(gist_weights.shape, gist_weights1.shape, gist_weights_mask.shape)
        atten_mask[reverse_cumsum(gist_idx_vector)>0] = 0
        gist_weights = gist_weights * atten_mask[:,None,:,None]
        gist_output = torch.matmul(gist_weights, gist_values.permute(1,0,2)) # [bsz, num_heads, q_len, head_dim]
        # print("gist_values", gist_values, gist_values.shape, self.layer_idx)
        # print("gist_output", gist_output, gist_output.shape, self.layer_idx)


        if self.training and gist_pool_idx is not None and self.layer_idx >= self.n_layers//2:
            eps = 1e-5
            gist_pool_idx = gist_pool_idx.to(atten_mask.device)

            p0_loss = -(gist_weights[...,0] + eps).log()
            p0_loss = ((p0_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
            # print("p0_loss", p0_loss)
            if p0_loss != 0:
                extra_loss["p0_loss"].append(p0_loss.cpu())


            gist_pool_idx[...,0] = 1
            mask = atten_mask[...,None].matmul(gist_pool_idx[:,None,:].to(atten_mask.dtype))
            # mask[...,0] = 1
            # print("mask", mask, mask.shape)

            pS_loss = -((gist_weights * mask[:,None,:,:]).sum(-1) + eps).log()
            pS_loss = ((pS_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
            # print("pS_loss", pS_loss)
            if pS_loss != 0:
                extra_loss["pS_loss"].append(pS_loss.cpu())


            #Entropy
            sparsity_loss = -gist_weights * (gist_weights+eps).log()
            sparsity_loss = ((sparsity_loss.sum(dim=-1) * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
            # print("sparsity_loss", sparsity_loss)
            if sparsity_loss != 0:
                extra_loss["sparsity_loss"].append(sparsity_loss.cpu())
        
            # print(p0_loss, pS_loss, sparsity_loss)



        if layer_past is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "partial_rotation_size": self.rotary_dim,
                "cache_position": cache_position,
            }
            key, value = layer_past.update(key, value, self.layer_idx, cache_kwargs)

        # compute self-attention: V x Softmax(QK^T)
        # attn_output, attn_weights = self._attn(query, key, value, causal_mask, head_mask)
        
        # Using sdpa attention
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query.device.type == "cuda" and causal_mask is not None:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(query=query, 
                                                                        key=key, 
                                                                        value=value, 
                                                                        attn_mask=causal_mask, 
                                                                        dropout_p=self.config.attn_pdrop,
                                                                        is_causal=False)

        attn_output = attn_output + gist_output

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, layer_past)
        # if output_attentions:
        #     outputs += (attn_weights,)

        return outputs  # a, present, (attentions)
    




class Edit_GPTJBlock(GPTJBlock):
    def __init__(self, config, layer_idx=None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.attn = Edit_GPTJAttention(config, layer_idx)
        self.ln_1 = LigerLayerNorm(config.n_embd, eps=config.layer_norm_epsilon, bias=True)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Cache] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        gist_pool: Optional[Dict[int, Dict]] = None, #[num_gist+1, num_key_value_head, head_dim]
        gist_idx_vector: Optional[torch.Tensor] = None, #[bsz, q_len]
        gist_pool_idx: Optional[torch.Tensor] = None, #[bsz, num_gist+1]
        topk: Optional[int] = None,
        extra_loss: Optional[Dict[str, List]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            gist_pool=gist_pool,
            gist_idx_vector=gist_idx_vector,
            gist_pool_idx=gist_pool_idx,
            topk=topk,
            extra_loss=extra_loss,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions)
    



class Edit_GPTJModel(GPTJModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.h = nn.ModuleList([Edit_GPTJBlock(config, layer_idx=i) for i in range(config.n_layer)])
        self.ln_f = LigerLayerNorm(self.embed_dim, eps=config.layer_norm_epsilon, bias=True)


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,     
        gist_pool_idx: Optional[torch.Tensor] = None,       
        topk: Optional[int] = 8,
        extra_loss: Optional[Dict[str, List]] = None,
        head_mask: Optional[torch.FloatTensor] = None,
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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

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

        seq_length = inputs_embeds.shape[1]
        if cache_position is None:
            past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        

        #gist location boolean vector
        gist_idx_vector = (input_ids == gist_token_ids)

        # print(position_ids)
        position_ids = alter_position_ids(gist_token_ids=gist_token_ids, input_ids=input_ids, origin_pos_ids=position_ids)
        # print(position_ids)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )


        attention_mask = attention_mask.to(inputs_embeds.dtype)
        col_segment = (reverse_cumsum(gist_idx_vector) > 0).to(inputs_embeds.dtype)
        # row_segment = ((gist_idx_vector.cumsum(-1).cumsum(-1) - 1) > 0).to(inputs_embeds.dtype)
        row_segment = col_segment.logical_not().to(inputs_embeds.dtype)
        mask1 = row_segment[...,None].matmul(col_segment[:,None,:])
        mask1[attention_mask[...,None].matmul(attention_mask[:,None,:]).logical_not()] = 0  # avoid changing the pad token value in causal mask 
        # print("mask1", mask1, mask1.shape)
        causal_mask[mask1[:,None,...].to(torch.bool)] = torch.finfo(inputs_embeds.dtype).min
        # causal_mask[mask1[:,None,...].to(torch.bool)] = torch.finfo(torch.float16).min
        # print("modified casual_mask", causal_mask, causal_mask.shape)



        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x num_attention_heads x N x N
        # head_mask has shape n_layer x batch x num_attention_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        hidden_states = inputs_embeds

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, seq_length)
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)
        output_shape = (-1, seq_length, hidden_states.size(-1))

        next_decoder_cache = None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)

                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if past_key_values is not None:
                    past_key_values.key_cache = past_key_values.key_cache.to(hidden_states.device)
                    past_key_values.value_cache = past_key_values.value_cache.to(hidden_states.device)

                # Ensure that attention_mask is always on the same device as hidden_states
                if causal_mask is not None:
                    causal_mask = causal_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    block.__call__,
                    hidden_states,
                    None,
                    (causal_mask, attention_mask),
                    gist_idx_vector,
                    gist_pool,
                    gist_pool_idx,
                    topk,
                    extra_loss,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    output_attentions,
                    cache_position,
                )
            else:
                outputs = block(
                    hidden_states=hidden_states,
                    layer_past=past_key_values,
                    attention_mask=(causal_mask, attention_mask),
                    gist_idx_vector=gist_idx_vector,
                    gist_pool=gist_pool,
                    gist_pool_idx=gist_pool_idx,
                    topk=topk,
                    extra_loss=extra_loss,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    cache_position=cache_position,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                next_decoder_cache = outputs[1]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states, all_self_attentions] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )




class Edit_GPTJForCausalLM(GPTJForCausalLM):
    def __init__(self, config,):
        super().__init__(config=config)
        self.transformer = Edit_GPTJModel(config)
        self.gist_pool = None
        self.gist_token_ids = config.vocab_size + 1
        self.gist_pool_idx = None
        self.topk = 8



    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, Tuple[Tuple[torch.Tensor]]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,
        gist_pool_idx: Optional[torch.Tensor] = None,
        topk: Optional[int] = None,
        extra_loss: Optional[Dict[str, List]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print(self.gist_pool, self.gist_token_ids, self.gist_pool_idx)
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            gist_pool=gist_pool if gist_pool is not None else self.gist_pool,
            gist_token_ids=gist_token_ids if gist_token_ids is not None else self.gist_token_ids,
            gist_pool_idx=gist_pool_idx if gist_pool_idx is not None else self.gist_pool_idx,
            topk=topk if topk is not None else self.topk,
            extra_loss=extra_loss,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        # make sure sampling in fp16 works correctly and
        # compute loss in fp32 to match with mesh-tf version
        # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )