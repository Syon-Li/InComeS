import torch
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
    LlamaMLP,
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
    AttentionMaskConverter,
    LLAMA_ATTENTION_CLASSES
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache



def reverse_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Cumulative sum from right to left.
    See https://github.com/pytorch/pytorch/issues/33520.
    """
    return x + torch.sum(x, dim=-1, keepdims=True) - torch.cumsum(x, dim=-1)


def make_mask_pre_first_gist(inputs: torch.Tensor, gist_token: int, dtype=torch.int64) -> torch.Tensor:
    """Returns a mask where all tokens prior to the first gist token are masked out.

    Args:
        inputs: a Tensor of input tokens where the last dimension is the sequence length.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask."""
    return ((inputs == gist_token).cumsum(-1) >= 1).type(dtype)


def make_mask_post_last_gist(inputs: torch.Tensor, gist_token: int, dtype=torch.int64) -> torch.Tensor:
    """Returns a mask where all tokens after the last gist token are masked out.

    Computes the same as mask_pre_first_gist_token, but reverses the sequence before and after the cumsum.

    Args:
        inputs: a Tensor of input tokens where the last dimension is the sequence length.
        gist_token: the integer id of the gist token.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask.
    """
    return (reverse_cumsum(inputs == gist_token) >= 1).type(dtype)


def make_gist_mask(inputs: torch.Tensor, gist_token: int, pad_token: int, dtype=torch.int64) -> torch.Tensor:
    """Creates a gist mask from supplied inputs and gist/pad tokens.

    Tokens after the last gist cannot attend to tokens prior to the first gist. Additionally, tokens *before*
    the last gist cannot attend to tokens *after* the last gist.

    The gist mask is broadcasted to 4D (with a singleton dim 1) for compatibility with multi-headed attention
    (where dim 1 is the head dimension).

    Args:
        inputs: a Tensor of shape (batch_size, seq_len) input tokens.
        gist_token: the integer id of the gist token.
        pad_token: the integer id of the pad token. inputs == pad_token are masked out.
        dtype: the dtype of the mask, default int64.
    Returns:
        The requested mask of shape (batch_size, 1, seq_len, seq_len)
    """
    # Attention mask for tokens before the last gist token.
    pre_gist_mask = make_mask_post_last_gist(inputs, gist_token, dtype=torch.bool)[:, None, None]
    # Attention mask for tokens after the last gist token.
    post_gist_mask = make_mask_pre_first_gist(inputs, gist_token, dtype=torch.bool)[:, None, None]
    # Construct time masks by permuting to time dimension.
    pre_gist_time_mask = pre_gist_mask.permute((0, 1, 3, 2))

    mask = torch.where(pre_gist_time_mask, pre_gist_mask, post_gist_mask)
    mask = mask & (inputs != pad_token)[:, None, None] # Mask out pad tokens.

    return mask.type(dtype)



def alter_position_ids(gist_token_ids: int, input_ids: torch.Tensor):
    conditions = ((input_ids == gist_token_ids).cumsum(-1) >= 1)
    ones = torch.ones(input_ids.shape).to(input_ids.device)
    zeros = torch.zeros(input_ids.shape).to(input_ids.device)
    position_ids_right = torch.where(conditions, ones, zeros).cumsum(-1) - 2
    position_ids_left = reverse_cumsum(torch.where(conditions.eq(False), -ones, zeros)) - 1
    position_ids = torch.where(conditions, position_ids_right, position_ids_left)
    return position_ids





class Edit_LlamaAttention(LlamaAttention):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.zero_gist_key = nn.parameter.Parameter(data=torch.randn((1, config.num_key_value_heads, config.hidden_size // config.num_attention_heads)))
        self.zero_gist_value = nn.parameter.Parameter(data=torch.zeros((1, config.num_key_value_heads, config.hidden_size // config.num_attention_heads)), requires_grad=False)


    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            gist_pool: Optional[Dict[int, Dict]] = None, #[num_gist+1, num_key_value_head, head_dim]
            gist_idx_vector: Optional[torch.Tensor] = None, #[bsz, q_len]
            gist_pool_idx: Optional[torch.Tensor] = None, #[bsz, num_gist+1]
            extra_loss: Optional[Dict[str, List]] = None,
            loss_weights: Optional[Dict] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
            
            bsz, q_len, _ = hidden_states.size()

            if self.config.pretraining_tp > 1:
                key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
                query_slices = self.q_proj.weight.split(
                    (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
                )
                key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
                value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

                query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
                query_states = torch.cat(query_states, dim=-1)

                key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
                key_states = torch.cat(key_states, dim=-1)

                value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
                value_states = torch.cat(value_states, dim=-1)

            else:
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


            past_key_value = getattr(self, "past_key_value", past_key_value)
            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


            #updating new gist
            new_gist_keys = key_states.transpose(1,2)[gist_idx_vector]
            new_gist_values = value_states.transpose(1,2)[gist_idx_vector]
            gist_pool[self.layer_idx]["keys"] = torch.concat([gist_pool[self.layer_idx]["keys"], new_gist_keys])
            gist_pool[self.layer_idx]["values"] = torch.concat([gist_pool[self.layer_idx]["values"], new_gist_values])

            #Add zero gist key and value
            gist_keys = torch.concat([self.zero_gist_key, gist_pool[self.layer_idx]["keys"]]) # [num_gist+1,key_value_head,head_dim]
            gist_values = torch.concat([self.zero_gist_value, gist_pool[self.layer_idx]["values"]])
            # print('gist key and value size', gist_keys.shape, gist_values.shape)


            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            gist_keys = gist_keys.repeat_interleave(self.num_key_value_groups, dim=1).transpose(0,1)
            gist_values = gist_values.repeat_interleave(self.num_key_value_groups, dim=1).transpose(0,1)


            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            gist_logits = torch.matmul(query_states, gist_keys.transpose(1, 2)) / math.sqrt(self.head_dim)

            causal_mask, atten_mask = attention_mask
            if causal_mask is not None:  # no matter the length, we just slice it
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
                



            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            # Calculate gist token attention
            gist_weights = nn.functional.softmax(gist_logits, dim=-1, dtype=torch.float32).to(query_states.dtype)
            atten_mask[reverse_cumsum(gist_idx_vector)>0] = 0
            gist_weights_mask = atten_mask[...,None].matmul(torch.ones(bsz, 1, gist_logits.shape[-1]).to(hidden_states.device))
            gist_weights = gist_weights * gist_weights_mask[:,None,...]
            gist_weights = nn.functional.dropout(gist_weights, p=self.attention_dropout, training=self.training)
            gist_output = torch.matmul(gist_weights, gist_values)
            # print(attn_output.shape, gist_output.shape)

            
            sparsity_loss_w, p0_loss_w, pS_loss_w = loss_weights["sparsity_loss_w"], loss_weights["p0_loss_w"], loss_weights["pS_loss_w"]
            eps = 0.1
            if self.training:
                if p0_loss_w>0:
                    p0_loss = -(gist_weights[...,0] + eps).log()
                    p0_loss = ((p0_loss * atten_mask[:,None,:]).sum(-1) / atten_mask[:,None,:].sum(-1)).mean()
                    extra_loss["p0_loss"].append(p0_loss.mul(p0_loss_w).item())

                if pS_loss_w>0:
                    # q_len_segment = ((gist_idx_vector.cumsum(-1).cumsum(-1) - 1) > 0).to(torch.float)
                    # mask = q_len_segment[...,None].matmul(gist_pool_idx[:,None,:])
                    mask = atten_mask[...,None].matmul(gist_pool_idx[:,None,:])
                    mask[...,0] = 1
                    # print("mask", mask, mask.shape)

                    pS_loss = -((gist_weights * mask[:,None,:,:]).sum(-1) + eps).log()
                    pS_loss = ((pS_loss * atten_mask[:,None,:]).sum(-1) / atten_mask[:,None,:].sum(-1)).mean()

                    extra_loss["pS_loss"].append(pS_loss.mul(pS_loss_w).item())

                if sparsity_loss_w>0:
                    #Entropy
                    sparsity_loss = -(gist_weights+eps) * (gist_weights+eps).log()
                    sparsity_loss = ((sparsity_loss.sum(dim=-1) * atten_mask[:,None,:]).sum(-1) / atten_mask[:,None,:].sum(-1)).mean()
                    extra_loss["sparsity_loss"].append(sparsity_loss.mul(sparsity_loss_w).item())
            
                # print(p0_loss, pS_loss, sparsity_loss)


            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # Only selected tokens are given gist information
            attn_output = attn_output + gist_output
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

            if self.config.pretraining_tp > 1:
                attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
                o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
                attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
            else:
                attn_output = self.o_proj(attn_output)

            if not output_attentions:
                attn_weights = None

            return attn_output, attn_weights, past_key_value, gist_pool, extra_loss




class Edit_LlamaSdpaAttention(Edit_LlamaAttention):

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            gist_pool: Optional[Dict[int, Dict]] = None, #[num_gist+1, num_key_value_head, head_dim]
            gist_idx_vector: Optional[torch.Tensor] = None, #[bsz, q_len]
            gist_pool_idx: Optional[torch.Tensor] = None, #[bsz, num_gist+1]
            extra_loss: Optional[Dict[str, List]] = None,
            loss_weights: Optional[Dict] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
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
                )

            bsz, q_len, _ = hidden_states.size()

            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)


            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            cos, sin = self.rotary_emb(value_states, position_ids)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

            
            #updating new gist
            new_gist_keys = key_states.transpose(1,2)[gist_idx_vector]
            new_gist_values = value_states.transpose(1,2)[gist_idx_vector]
            gist_pool[self.layer_idx]["keys"] = torch.concat([gist_pool[self.layer_idx]["keys"], new_gist_keys])
            gist_pool[self.layer_idx]["values"] = torch.concat([gist_pool[self.layer_idx]["values"], new_gist_values])

            #Add zero gist key and value
            gist_keys = torch.concat([self.zero_gist_key, gist_pool[self.layer_idx]["keys"]]) # [num_gist+1,key_value_head,head_dim]
            gist_values = torch.concat([self.zero_gist_value, gist_pool[self.layer_idx]["values"]])


            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            causal_mask, atten_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

            gist_keys = gist_keys.repeat_interleave(self.num_key_value_groups, dim=1).transpose(0,1)
            gist_values = gist_values.repeat_interleave(self.num_key_value_groups, dim=1).transpose(0,1)

            gist_logits = torch.matmul(query_states, gist_keys.transpose(1, 2)) / math.sqrt(self.head_dim)
            atten_mask[reverse_cumsum(gist_idx_vector)>0] = 0
            # gist_logits_mask = atten_mask[...,None].matmul(torch.ones(bsz, 1, gist_logits.shape[-1]).to(hidden_states.device)) * torch.finfo(hidden_states.dtype).min
            gist_weights_mask = atten_mask[...,None].matmul(torch.ones(bsz, 1, gist_logits.shape[-1]).to(hidden_states.device))

            #Mask padding tokens and all tokens preceding gist tokens (including itself)
            # gist_logits = gist_logits + gist_logits_mask[:,None,...]
            
            # Calculate gist token attention
            gist_weights = nn.functional.softmax(gist_logits, dim=-1, dtype=torch.float32).to(query_states.dtype) # [bsz, num_heads, q_len, gist_num]
            gist_weights = nn.functional.dropout(gist_weights, p=self.attention_dropout, training=self.training)
            # print("gist_weight", gist_weights.shape)
            gist_weights = gist_weights * gist_weights_mask[:,None,...]    
            gist_output = torch.matmul(gist_weights, gist_values) # [bsz, num_heads, q_len, head_dim]
            # print("gist_output", gist_output, gist_output.shape)

            
            
            sparsity_loss_w, p0_loss_w, pS_loss_w = loss_weights["sparsity_loss_w"], loss_weights["p0_loss_w"], loss_weights["pS_loss_w"]
            eps = 1e-5
            if self.training:
                if p0_loss_w>0:
                    p0_loss = -(gist_weights[...,0] + eps).log()
                    p0_loss = ((p0_loss * atten_mask[:,None,:]).sum(-1) / atten_mask[:,None,:].sum(-1)).mean()
                    extra_loss["p0_loss"].append(p0_loss.mul(p0_loss_w).item())

                if pS_loss_w>0:
                    # q_len_segment = ((gist_idx_vector.cumsum(-1).cumsum(-1) - 1) > 0).to(torch.float)
                    # mask = q_len_segment[...,None].matmul(gist_pool_idx[:,None,:])
                    mask = atten_mask[...,None].matmul(gist_pool_idx[:,None,:])
                    mask[...,0] = 1
                    # print("mask", mask, mask.shape)

                    pS_loss = -((gist_weights * mask[:,None,:,:]).sum(-1) + eps).log()
                    pS_loss = ((pS_loss * atten_mask[:,None,:]).sum(-1) / atten_mask[:,None,:].sum(-1)).mean()

                    extra_loss["pS_loss"].append(pS_loss.mul(pS_loss_w).item())

                if sparsity_loss_w>0:
                    #Entropy
                    sparsity_loss = -(gist_weights+eps) * (gist_weights+eps).log()
                    sparsity_loss = ((sparsity_loss.sum(dim=-1) * atten_mask[:,None,:]).sum(-1) / atten_mask[:,None,:].sum(-1)).mean()
                    extra_loss["sparsity_loss"].append(sparsity_loss.mul(sparsity_loss_w).item())
            
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
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
            # print("pre_attn_output", attn_output, attn_output.shape)
            attn_output = attn_output + gist_output
            # print("after_attn_output", attn_output, attn_output.shape)
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, -1)

            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value, gist_pool, extra_loss
    





LLAMA_ATTENTION_CLASSES["eager"] = Edit_LlamaAttention
LLAMA_ATTENTION_CLASSES["sdpa"] = Edit_LlamaSdpaAttention




class Edit_LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_idx_vector: Optional[torch.Tensor] = None,    
        gist_pool_idx: Optional[torch.Tensor] = None,    
        extra_loss: Optional[Dict[str, List]] = None,
        loss_weights: Optional[Dict] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
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
        hidden_states, self_attn_weights, present_key_value, gist_pool, extra_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            gist_idx_vector=gist_idx_vector,
            gist_pool=gist_pool,
            gist_pool_idx=gist_pool_idx,
            extra_loss=extra_loss,
            loss_weights=loss_weights,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
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

        return outputs, gist_pool, extra_loss
   



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
        past_seen_tokens: int,
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
        # past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        # using_static_cache = isinstance(past_key_values, StaticCache)

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


        # dtype, device = input_tensor.dtype, input_tensor.device
        device = input_tensor.device
        dtype = torch.float16
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, torch.Tensor)
            else past_seen_tokens + sequence_length + 1
        )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
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
        extra_loss: Optional[Dict[str, List]] = None,
        loss_weights: Optional[Dict] = None,
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

        past_seen_tokens = 0
        if use_cache:  # kept for BC (cache positions)
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is a required argument when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        #gist location boolean vector
        gist_idx_vector = (input_ids == gist_token_ids)


        # print(position_ids)
        position_ids = alter_position_ids(gist_token_ids=gist_token_ids, input_ids=input_ids)
        # print(position_ids)

        # print("self.config._attn_implementation", self.config._attn_implementation)
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        # print("causal_mask:", causal_mask)


        col_segment = (reverse_cumsum(gist_idx_vector) > 0).to(torch.float)
        row_segment = ((gist_idx_vector.cumsum(-1).cumsum(-1) - 1) > 0).to(torch.float)
        mask1 = row_segment[...,None].matmul(col_segment[:,None,:])
        # print("mask1", mask1, mask1.shape)
        causal_mask[mask1[:,None,...].to(torch.bool)] = torch.finfo(torch.float16).min
        # print(inputs_embeds.dtype)
        # print("modified casual_mask", causal_mask, causal_mask.shape)
    

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None


        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs, gist_pool, extra_loss = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    attention_mask=(causal_mask, attention_mask),
                    gist_idx_vector=gist_idx_vector,
                    gist_pool=gist_pool,
                    gist_pool_idx=gist_pool_idx,
                    extra_loss=extra_loss,
                    loss_weights=loss_weights,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            else:
                # print(hidden_states, hidden_states.shape)
                layer_outputs, gist_pool, extra_loss = decoder_layer(
                    hidden_states,
                    attention_mask=(causal_mask, attention_mask),
                    gist_idx_vector=gist_idx_vector,
                    gist_pool=gist_pool,
                    gist_pool_idx=gist_pool_idx,
                    extra_loss=extra_loss,
                    loss_weights=loss_weights,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
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

        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache() if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), gist_pool, extra_loss





class Edit_LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = Edit_LlamaModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,
        gist_pool_idx: Optional[torch.Tensor] = None,
        loss_weights: Optional[Dict] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
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

        extra_loss = {"sparsity_loss":[], "p0_loss":[], "pS_loss":[]}
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, gist_pool, extra_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            gist_pool=gist_pool,
            gist_token_ids=gist_token_ids,
            gist_pool_idx=gist_pool_idx,
            extra_loss=extra_loss,
            loss_weights=loss_weights,
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
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        loss_set = {}
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            loss_set.update({"CrossEntropy_loss": loss.item()})
            for k,v in extra_loss.items():
                loss_k = torch.tensor(v).to(loss.device).mean()
                loss_set.update({k:loss_k.item()})
                loss = loss + loss_k

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), gist_pool, loss_set
    
