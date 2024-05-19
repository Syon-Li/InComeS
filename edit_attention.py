import torch
import warnings
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union, Dict
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaModel,
    LlamaForCausalLM,
    LlamaRMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
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

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            gist_activations: Optional[Dict[int, Dict]] = None,
            gist_idx_vector: Optional[torch.Tensor] = None,
            gist_pool_idx: Optional[torch.Tensor] = None,
            extra_loss: Optional[Dict[str, List]] = None,
            loss_weights: Optional[Tuple] = None,
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
            # new_gist_keys = key_states[F.one_hot(gist_token_idxs, num_classes=q_len).to(torch.bool)]
            # new_gist_values = value_states[F.one_hot(gist_token_idxs, num_classes=q_len).to(torch.bool)]
            new_gist_keys = key_states.transpose(1,2)[gist_idx_vector].detach()
            new_gist_values = value_states.transpose(1,2)[gist_idx_vector].detach()
            gist_activations[self.layer_idx]["keys"] = torch.vstack([gist_activations[self.layer_idx]["keys"], new_gist_keys])
            gist_activations[self.layer_idx]["values"] = torch.vstack([gist_activations[self.layer_idx]["values"], new_gist_values])

            #Gist activations
            gist_keys = gist_activations[self.layer_idx]["keys"]
            gist_values = gist_activations[self.layer_idx]["values"]
            # print('gist_pool size', gist_keys.shape, gist_values.shape)
            # num_gist_thresh = 20
            # if gist_keys.shape[0] > num_gist_thresh:
            #     gist_keys[:-num_gist_thresh] = gist_keys[:-num_gist_thresh].detach()
            #     gist_values[:-num_gist_thresh] = gist_values[:-num_gist_thresh].detach()
            bsz_gist_keys = torch.stack([gist_keys]*bsz, dim=0)
            bsz_gist_values = torch.stack([gist_values]*bsz, dim=0)
            bsz_gist_keys = bsz_gist_keys.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            bsz_gist_values = bsz_gist_values.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            # print(key_states.shape, gist_layer_activa.shape)



            if past_key_value is not None:
                # sin and cos are specific to RoPE models; cache_position needed for the static cache
                cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            bsz_gist_keys = repeat_kv(bsz_gist_keys, self.num_key_value_groups)
            bsz_gist_values = repeat_kv(bsz_gist_values, self.num_key_value_groups)

            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
            gist_weights = torch.matmul(query_states, bsz_gist_keys.transpose(2, 3)) / math.sqrt(self.head_dim)


            if attention_mask is not None:  # no matter the length, we just slice it
                causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
                attn_weights = attn_weights + causal_mask
                



            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output = torch.matmul(attn_weights, value_states)

            # Calculate gist token attention
            gist_weights = nn.functional.softmax(gist_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            gist_weights = nn.functional.dropout(gist_weights, p=self.attention_dropout, training=self.training)
            gist_output = torch.matmul(gist_weights, bsz_gist_values)
            # print(attn_output.shape, gist_output.shape)

            
            sparsity_loss_w, p0_loss_w, pS_loss_w = loss_weights
            if self.training:
                if p0_loss_w>0:
                    p0_loss = -gist_weights[...,0].log().mean()
                    extra_loss["p0_loss"].append(p0_loss.mul(p0_loss_w))

                if pS_loss_w>0:
                    q_len_index = ((gist_idx_vector.cumsum(-1).cumsum(-1) - 1) > 0).reshape(-1, bsz)
                    
                    mask = q_len_index.repeat_interleave(gist_weights.shape[-1], dim=1).reshape(bsz, q_len, -1) * gist_pool_idx.repeat_interleave(q_len, dim=0).reshape(bsz, q_len, -1)
                    # print(gist_pool_idx.repeat_interleave(q_len, dim=0).reshape(bsz, q_len, -1))
                    mask = mask.repeat_interleave(self.num_heads, dim=1).reshape(gist_weights.shape)
                    # print(mask, mask.shape)

                    # pS_loss = -gist_weights[gist_weights_mask.transpose(2,3).to(torch.bool)].log().sum(dim=0).mean()
                    # pS_loss = -gist_weights[gist_weights_mask.transpose(2,3).to(torch.bool)].log().mean()
                    pS_loss = -gist_weights[mask.to(torch.bool)].log().mean()
                    extra_loss["pS_loss"].append(pS_loss.mul(pS_loss_w))

                if sparsity_loss_w>0:
                    #Entropy
                    sparsity_loss = -(gist_weights * gist_weights.log()).sum(dim=-1).mean()
                    extra_loss["sparsity_loss"].append(sparsity_loss.mul(sparsity_loss_w))
            
                # print(sparsity_loss, p0_loss, pS_loss)


            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            attn_output += gist_output
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

            return attn_output, attn_weights, past_key_value, gist_activations, extra_loss




class Edit_LlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.self_attn = Edit_LlamaAttention(config, layer_idx=layer_idx)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        gist_activations: Optional[Dict[int, torch.Tensor]] = None,
        gist_idx_vector: Optional[torch.Tensor] = None,    
        gist_pool_idx: Optional[torch.Tensor] = None,    
        extra_loss: Optional[Dict[str, List]] = None,
        loss_weights: Optional[Tuple] = None,
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
        hidden_states, self_attn_weights, present_key_value, gist_activations, extra_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            gist_idx_vector=gist_idx_vector,
            gist_activations=gist_activations,
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

        return outputs, gist_activations, extra_loss
   



class Edit_LlamaModel(LlamaModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList([Edit_LlamaDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)])
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        gist_activations: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,     
        gist_pool_idx: Optional[torch.Tensor] = None,       
        extra_loss: Optional[Dict[str, List]] = None,
        loss_weights: Optional[Tuple] = None,
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
            # logger.warning_once(
            #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            # )
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


        #Extra loss and their weights for gist training
        loss_weights = (1,1,1)
        gist_idx_vector = (input_ids == gist_token_ids)


        # print(position_ids)
        position_ids = alter_position_ids(gist_token_ids=gist_token_ids, input_ids=input_ids)
        # print(position_ids)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_seen_tokens)
        # print(causal_mask.shape)

        # gist_mask = make_gist_mask(input_ids, gist_token=gist_token_ids, pad_token=2)
        # causal_gist_mask = torch.where(gist_mask>0, causal_mask, torch.finfo(inputs_embeds.dtype).min)
        # row_index = (gist_idx_vector.cumsum(-1) - 1) > 0
        # mask1 = row_index.repeat_interleave(input_ids.shape[1], dim=0) * gist_idx_vector.repeat_interleave(input_ids.shape[1], dim=0)
        # causal_gist_mask[mask1.reshape(causal_gist_mask.shape)] = torch.finfo(inputs_embeds.dtype).min
        # print(causal_gist_mask)


        col_index = (reverse_cumsum(reverse_cumsum(gist_idx_vector)) - 1 > 0)
        row_index = ((gist_idx_vector.cumsum(-1).cumsum(-1) - 1) > 0).reshape(-1, input_ids.shape[0])
        # print(col_index)
        # print(row_index)
        mask1 = row_index.repeat_interleave(input_ids.shape[1], dim=1).reshape(causal_mask.shape) * col_index.repeat_interleave(input_ids.shape[1], dim=0).reshape(causal_mask.shape)
        causal_mask[mask1] = torch.finfo(inputs_embeds.dtype).min
        # print(causal_mask, causal_mask.shape)

        # try:
        #     gist_mask = make_gist_mask(input_ids, gist_token=gist_token_ids, pad_token=2)
        #     # print(gist_mask.shape)
        #     # gist_mask = (gist_mask.eq(0)).to(inputs_embeds.dtype) * torch.finfo(inputs_embeds.dtype).min
        #     causal_gist_mask = torch.where(gist_mask>0, causal_mask, torch.finfo(inputs_embeds.dtype).min)
        #     causal_gist_mask[:, gist_idx_vector] = torch.finfo(inputs_embeds.dtype).min
        #     print(causal_gist_mask)
        # except:
        #     causal_gist_mask = causal_mask
    

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
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs, gist_activations, extra_loss = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    gist_idx_vector=gist_idx_vector,
                    gist_activations=gist_activations,
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
        ), gist_activations, extra_loss





class Edit_LlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.model = Edit_LlamaModel(config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        gist_activations: Optional[Dict[int, torch.Tensor]] = None,
        gist_token_ids: Optional[int] = None,
        gist_pool_idx: Optional[torch.Tensor] = None,
        # loss_weights: Optional[torch.Tensor] = None,
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
        loss_weights = (1,1,1)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, gist_activations, extra_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            gist_activations=gist_activations,
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
            loss += torch.tensor(extra_loss["sparsity_loss"]).mean()
            loss += torch.tensor(extra_loss["p0_loss"]).mean()
            loss += torch.tensor(extra_loss["pS_loss"]).mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), gist_activations
    