#  transformer v4.45.2

import torch
import copy
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union, Dict
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2SdpaAttention,
    Qwen2FlashAttention2,
    _flash_attention_forward,
    Qwen2DecoderLayer,
    Qwen2Config,
    Qwen2Model,
    Qwen2ForCausalLM,
    GenerationMixin,
    apply_rotary_pos_emb,
    _prepare_4d_causal_attention_mask_with_cache_position,
    repeat_kv,
    AttentionMaskConverter,
)
# from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils import logging, is_torchdynamo_compiling
from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss, LigerFusedLinearJSD
from utils import reverse_cumsum, alter_position_ids


logger = logging.get_logger(__name__)


class Edit_Qwen2SdpaAttention(Qwen2SdpaAttention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.zero_gist_key = nn.parameter.Parameter(data=torch.randn(1, self.num_key_value_heads, self.head_dim))
        self.n_layers = config.num_hidden_layers
        if layer_idx < self.n_layers//2:
            self.zero_gist_key.requires_grad = False
        else:
            self.zero_gist_key.requires_grad = True


    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        gist_pool: Optional[Dict[int, Dict]] = None, #[num_gist+1, num_key_value_head, head_dim]
        gist_idx_vector: Optional[torch.Tensor] = None, #[bsz, q_len]
        gist_pool_idx: Optional[torch.Tensor] = None, #[bsz, num_gist+1]
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        gist_pool = copy.copy(gist_pool) # in favor of the gradient checkpointing
        gist_pool['keys'] = gist_pool['keys'].to(query_states.device)
        gist_pool["values"] = gist_pool["values"].to(query_states.device)

        # updating new gist
        if self.layer_idx >= self.n_layers//2 and gist_idx_vector.any().item():
            new_gist_keys = key_states.transpose(1,2)[gist_idx_vector]
            new_gist_values = value_states.transpose(1,2)[gist_idx_vector]                               
            gist_pool["keys"] = torch.concat([gist_pool["keys"], new_gist_keys]).to(query_states.device)
            gist_pool["values"] = torch.concat([gist_pool["values"], new_gist_values]).to(query_states.device)

        raw_gist_keys = gist_pool["keys"]
        raw_gist_values = gist_pool["values"]

    
        # Add zero gist key and value
        zero_values = torch.zeros(1, self.num_key_value_heads, self.head_dim, device=query_states.device)
        gist_keys = torch.concat([self.zero_gist_key, raw_gist_keys]).to(query_states.dtype) # [num_gist+1,key_value_head,head_dim]
        gist_values = torch.concat([zero_values, raw_gist_values]).to(query_states.dtype)   



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

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask, atten_mask = attention_mask
        if causal_mask is not None:  # no matter the length, we just slice it
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]


        num_gist = gist_keys.shape[0]
        gist_keys = gist_keys[:,:,None,:].expand(-1, -1, self.num_key_value_groups, -1).reshape(num_gist, self.num_heads, self.head_dim)
        gist_values = gist_values[:,:,None,:].expand(-1, -1, self.num_key_value_groups, -1).reshape(num_gist, self.num_heads, self.head_dim)
        gist_logits = torch.matmul(query_states.to(gist_keys.dtype), gist_keys.permute(1, 2, 0)) / math.sqrt(self.head_dim) # [bsz, num_heads, q_len, num_gist]

        if not self.training and self.layer_idx >= self.n_layers//2 and gist_keys.shape[0]>10:
            gist_logits = gist_logits / 0.45
        gist_weights = nn.functional.softmax(gist_logits, dim=-1, dtype=torch.float32).to(query_states.dtype) # [bsz, num_heads, q_len, gist_num]
        gist_weights = nn.functional.dropout(gist_weights, p=self.attention_dropout, training=self.training)
        atten_mask = atten_mask.to(key_states.dtype).clone()
        atten_mask[reverse_cumsum(gist_idx_vector)>0] = 0
        gist_weights = gist_weights * atten_mask[:,None,:,None]
        gist_output = torch.matmul(gist_weights, gist_values.permute(1, 0, 2)) # [bsz, num_heads, q_len, head_dim]



        extra_loss = {}
        if self.training and gist_pool_idx is not None and self.layer_idx >= self.n_layers//2:
            with torch.no_grad():
                eps = 1e-5

                p0_loss = -(gist_weights[...,0] + eps).log()
                p0_loss = ((p0_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()

                if torch.isnan(p0_loss).item():
                    print("gist_weight with NAN", gist_weights, gist_weights.sum(-1), gist_weights.shape)
                    print("attention_mask", atten_mask, atten_mask.sum(-1))
                
                # print("p0_loss", p0_loss)
                extra_loss.update({"p0_loss": p0_loss})


                pS_loss = -((gist_weights * atten_mask[:,None,:,None] * gist_pool_idx[:,None,None,:]).sum(-1) + eps).log()
                pS_loss = ((pS_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
                # print("pS_loss", pS_loss)
                extra_loss.update({"pS_loss": pS_loss})


                # Entropy
                sparsity_loss = -gist_weights * (gist_weights+eps).log()
                sparsity_loss = ((sparsity_loss.sum(dim=-1) * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
                # print("sparsity_loss", sparsity_loss)
                extra_loss.update({"sparsity_loss": sparsity_loss})
            
                # print(p0_loss, pS_loss, sparsity_loss)



        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output + gist_output

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value, gist_pool, extra_loss





class Edit_Qwen2FlashAttention2(Qwen2FlashAttention2):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.zero_gist_key = nn.parameter.Parameter(data=torch.randn(1, self.num_key_value_heads, self.head_dim))
        self.n_layers = config.num_hidden_layers
        if layer_idx < self.n_layers//2:
            self.zero_gist_key.requires_grad = False
        else:
            self.zero_gist_key.requires_grad = True


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gist_pool: Optional[Dict[int, Dict]] = None, #[num_gist+1, num_key_value_head, head_dim]
        gist_idx_vector: Optional[torch.Tensor] = None, #[bsz, q_len]
        gist_pool_idx: Optional[torch.Tensor] = None, #[bsz, num_gist+1]
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        gist_pool = copy.copy(gist_pool) # in favor of the gradient checkpointing
        gist_pool['keys'] = gist_pool['keys'].to(query_states.device)
        gist_pool["values"] = gist_pool["values"].to(query_states.device)

        # updating new gist
        if self.layer_idx >= self.n_layers//2 and gist_idx_vector.any().item():
        # if gist_idx_vector.any().item():
            new_gist_keys = key_states.transpose(1,2)[gist_idx_vector]
            new_gist_values = value_states.transpose(1,2)[gist_idx_vector]                               
            gist_pool["keys"] = torch.concat([gist_pool["keys"], new_gist_keys]).to(query_states.device)
            gist_pool["values"] = torch.concat([gist_pool["values"], new_gist_values]).to(query_states.device)

        raw_gist_keys = gist_pool["keys"]
        raw_gist_values = gist_pool["values"]   
    
        # Add zero gist key and value
        zero_values = torch.zeros(1, self.num_key_value_heads, self.head_dim, device=query_states.device)
        gist_keys = torch.concat([self.zero_gist_key, raw_gist_keys]).to(query_states.dtype) # [num_gist+1,key_value_head,head_dim]
        gist_values = torch.concat([zero_values, raw_gist_values]).to(query_states.dtype)   

        causal_mask, atten_mask = attention_mask


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

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            kv_seq_len = key_states.shape[-2] + cache_position[0]
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if causal_mask is not None:
                    causal_mask = causal_mask[:, slicing_tokens:]
                    causal_mask = torch.cat([causal_mask, torch.ones_like(causal_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)


        num_gist = gist_keys.shape[0]
        gist_keys = gist_keys[:,:,None,:].expand(-1, -1, self.num_key_value_groups, -1).reshape(num_gist, self.num_heads, self.head_dim)
        gist_values = gist_values[:,:,None,:].expand(-1, -1, self.num_key_value_groups, -1).reshape(num_gist, self.num_heads, self.head_dim)
        gist_logits = torch.matmul(query_states.to(gist_keys.dtype), gist_keys.permute(1, 2, 0)) / math.sqrt(self.head_dim) # [bsz, num_heads, q_len, num_gist]


        if not self.training and self.layer_idx >= self.n_layers//2 and gist_keys.shape[0]>10:
        # if not self.training:
            gist_logits = gist_logits / 0.45
        gist_weights = nn.functional.softmax(gist_logits, dim=-1, dtype=torch.float32).to(query_states.dtype) # [bsz, num_heads, q_len, gist_num]
        gist_weights = nn.functional.dropout(gist_weights, p=self.attention_dropout, training=self.training)
        atten_mask = atten_mask.to(key_states.dtype).clone()
        atten_mask[reverse_cumsum(gist_idx_vector)>0] = 0
        gist_weights = gist_weights * atten_mask[:,None,:,None]
        gist_output = torch.matmul(gist_weights, gist_values.permute(1, 0, 2)) # [bsz, num_heads, q_len, head_dim]


        extra_loss = {}
        if self.training and gist_pool_idx is not None and self.layer_idx >= self.n_layers//2:
        # if self.training and gist_pool_idx is not None:   
            with torch.no_grad():
                eps = 1e-5

                p0_loss = -(gist_weights[...,0] + eps).log()
                p0_loss = ((p0_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
                if torch.isnan(p0_loss).item():
                    print("gist_weight with NAN", gist_weights, gist_weights.sum(-1), gist_weights.shape)
                    print("attention_mask", atten_mask, atten_mask.sum(-1))
                extra_loss.update({"p0_loss": p0_loss})

                pS_loss = -((gist_weights * atten_mask[:,None,:,None] * gist_pool_idx[:,None,None,:]).sum(-1) + eps).log()
                pS_loss = ((pS_loss * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
                extra_loss.update({"pS_loss": pS_loss})

                # Entropy
                sparsity_loss = -gist_weights * (gist_weights+eps).log()
                sparsity_loss = ((sparsity_loss.sum(dim=-1) * atten_mask[:,None,:]).sum(-1) / (atten_mask[:,None,:].sum(-1) + eps)).mean()
                # print("sparsity_loss", sparsity_loss)
                extra_loss.update({"sparsity_loss": sparsity_loss})
            
                # print(p0_loss, pS_loss, sparsity_loss)



        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            causal_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )
        attn_output = attn_output + gist_output.transpose(1,2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value, gist_pool, extra_loss
    




Edit_QWEN2_ATTENTION_CLASSES = {
    "flash_attention_2": Edit_Qwen2FlashAttention2,
    "sdpa": Edit_Qwen2SdpaAttention,
}





class Edit_Qwen2DecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Edit_QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.layer_idx = layer_idx


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_idx_vector: Optional[torch.Tensor] = None,    
        gist_pool_idx: Optional[torch.Tensor] = None,   
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
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value, gist_pool, extra_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            gist_idx_vector=gist_idx_vector,
            gist_pool=gist_pool,
            gist_pool_idx=gist_pool_idx,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
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
    



class Edit_Qwen2Model(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList([Edit_Qwen2DecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,     
        gist_pool_idx: Optional[torch.Tensor] = None,
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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

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

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        #gist location boolean vector
        gist_idx_vector = (input_ids == gist_token_ids)

        # print("origin position_ids", position_ids, sep="\n")
        # position_ids = alter_position_ids(gist_token_ids=gist_token_ids, input_ids=input_ids, origin_pos_ids=position_ids)
        # print("new_pos_ids", position_ids, sep="\n")

        # print("self.config._attn_implementation", self.config._attn_implementation)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        ) # [bsz, 1, q_len, q_len]
        # print("causal_mask:", causal_mask, causal_mask.shape)


        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs, gist_layer_pool, extra_loss_rnt = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    (causal_mask, attention_mask),
                    gist_pool[layer_idx],
                    gist_idx_vector, 
                    gist_pool_idx,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs, gist_layer_pool, extra_loss_rnt = decoder_layer(
                    hidden_states,
                    attention_mask=(causal_mask, attention_mask),
                    gist_idx_vector=gist_idx_vector,
                    gist_pool=gist_pool[layer_idx],
                    gist_pool_idx=gist_pool_idx,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            gist_pool[layer_idx] = gist_layer_pool
            if extra_loss is not None:
                for k,v in extra_loss_rnt.items():
                    if k not in extra_loss.keys():
                        extra_loss.update({k:[]})
                    extra_loss[k].append(v)

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




class Edit_Qwen2ForCausalLM(Qwen2ForCausalLM, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = Edit_Qwen2Model(config)
        self.gist_pool = {}
        for i in range(config.num_hidden_layers):
            self.gist_pool.update({i:{"keys":torch.tensor([]), "values":torch.tensor([])}})
        self.gist_token_ids = config.vocab_size + 1
        self.gist_pool_idx = None


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        gist_pool: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,
        gist_pool_idx: Optional[torch.Tensor] = None,
        token_weights: Optional[torch.Tensor] = None,
        teacher_input: Optional[torch.Tensor] = None,
        teacher_weight: Optional[torch.Tensor] = None,
        extra_loss: Optional[Dict[str, List]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        T_labels: Optional[torch.LongTensor] = None,
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

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

        >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

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
        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # TODO: remove the float() operation in v4.46
        # print("lm_head", self.lm_head.weight)

        loss = None
        logits = None

        if self.training and (labels is not None):
            shift_hidden_states = hidden_states[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_T_labels = T_labels[..., 1:].contiguous()
            # shift_teacher_input = teacher_input[..., :-1, :].contiguous().to(hidden_states.device)

            # flatten tokens
            shift_hidden_states = shift_hidden_states.view(-1, self.config.hidden_size)
            shift_labels = shift_labels.view(-1)
            shift_T_labels = shift_T_labels.view(-1)
            # shift_teacher_input = shift_teacher_input.view(-1, self.config.hidden_size)

            lce = LigerFusedLinearCrossEntropyLoss(reduction="none")
            fused_kiv = LigerFusedLinearJSD(jsd_beta=0)

            mask = (shift_labels != -100)
            T_mask = (shift_T_labels != -100)

            ce_loss = lce(lin_weight=self.lm_head.weight, _input=shift_hidden_states[T_mask], target=shift_labels[T_mask])        
            klloss = fused_kiv(student_input=shift_hidden_states[T_mask], 
                                student_weight=self.lm_head.weight, 
                                teacher_input=teacher_input,
                                teacher_weight=teacher_weight.to(hidden_states.device),
                                shift_labels=None,
                                )
            w = token_weights if token_weights is not None else 1
            # wce_loss = (ce_loss * w).mean()
            wce_loss = (ce_loss * w).sum() / (w!=0).sum()
            loss = wce_loss + klloss
        else:
            logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])
            if labels is not None:
                # Upcast to float if we need to compute the loss to avoid potential precision issues
                logits = logits.float()
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=(loss, ce_loss.mean(), wce_loss, klloss) if self.training and (labels is not None) else loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
