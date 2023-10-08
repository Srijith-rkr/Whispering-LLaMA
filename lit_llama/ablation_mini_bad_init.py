"""Implementation of the paper:

LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention
https://arxiv.org/abs/2303.16199
"""
# mypy: ignore-errors
from dataclasses import dataclass
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

import lit_llama.model as llama
from lit_llama.model import build_rope_cache, apply_rope, RMSNorm, MLP, KVCache, RoPECache
# You implement only LLAMA, Block and MHA , you import MLP and RMSNorm from the main model 


@dataclass
class LLaMAConfig(llama.LLaMAConfig):
    adapter_prompt_length: int = 10
    adapter_start_layer: int = 2


class CausalSelfAttention(nn.Module):
    """A modification of `lit_llama.model.CausalSelfAttention` that adds the attention
    over the adaption prompt."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

        if block_idx >= config.adapter_start_layer:
            # adapter embedding layer
            self.adapter_wte = nn.Embedding(config.adapter_prompt_length, config.n_embd)
            # gate for adaption
            self.gating_factor = torch.nn.Parameter(torch.zeros(1))
            self.rms_gate = RMSNorm(config.n_embd)

            self.projection_rms_key = RMSNorm(1280)
            self.projection_key_matrix_down = torch.nn.Parameter(torch.randn([1280,1280//16]))
            self.projection_key_matrix_up =  torch.nn.Parameter(torch.randn([1280//16,1280]))#nn.Linear( 32, 32, bias=False) # torch.nn.Parameter(torch.randn([20,32]))

            self.projection_rms_value = RMSNorm(1280)
            # self.projection_value_matrix_down = torch.nn.Parameter(torch.zeros([1280,1280//8]).fill_diagonal_(1))#nn.Linear( 128, 128, bias=False) #torch.nn.Parameter(torch.randn([64,128]))
            # self.projection_value_matrix_up =  torch.nn.Parameter(torch.zeros([1280//8,1280]).fill_diagonal_(1))#nn.Linear( 32, 32, bias=False) # torch.nn.Parameter(torch.randn([20,32]))

            self.projection_query_matrix_128to128 = torch.nn.Parameter(torch.zeros([128,128]).fill_diagonal_(1)) #nn.Linear( 128, 128, bias=False) 
            self.projection_query_matrix_32to32 =   torch.nn.Parameter(torch.zeros([32,32]).fill_diagonal_(1))#nn.Linear( 32, 32, bias=False)

            self.projection_gating_factor = torch.nn.Parameter(torch.ones(1))
            #BOMB
            n_state = 64*20 # 20 heads and 64 word size
            self.whisper_key = nn.Linear(n_state, n_state, bias=False)
            self.whisper_value = nn.Linear(n_state, n_state)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        # These 3 lines below are extra
        self.block_idx = block_idx
        self.adapter_prompt_length = config.adapter_prompt_length
        self.adapter_start_layer = config.adapter_start_layer

    def forward(
        self,
        x: torch.Tensor,
        audio_features: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None, w_kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q,k,v denotes lammas's embeddings and key and value denote whisper's embeddings
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size) 
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        # break the matrix into headsize for whisper key and value also
        # key = key.view(B, 1500, 32, 128) # 1500 = audio context; 20 = num_heads; 64 = head_size
        # value = value.view(B, 1500, 32, 128) 

        # applying positional embeddings
        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        # swapping head out of the matrix multiplication. you want the shape batch, heads, context, latent-vector
        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)
        # key = key.transpose(1,2)
        # value = value.transpose(1,2)

        if kv_cache is not None: # only gets activated during inference
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)

            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        # The if statement is extra
        if self.block_idx >= self.adapter_start_layer:
            if adapter_kv_cache is not None: # passed only during inference
                ak, av = adapter_kv_cache
            else:
                prefix = self.adapter_wte.weight.reshape(1, self.adapter_prompt_length, self.n_embd) # 10,4096 -> 1,10,4096
                aT = prefix.size(1) # aT is adapter prompt length. T is used to denote prompt length hence aT
                _ , ak, av = self.c_attn(self.rms_gate(prefix)).split(self.n_embd, dim=2)
                # aq was initially not used (was _) but while combining whisper embedding, the key and value are fixed , you need a learnable query - hence used whispers query. (similar to adapters, they had frozen query and learnable key and value)
                # later realised the T dimension will be 10 and not the variable T if we use ak; hence went back to using Lallma query
                # aq = aq.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2) 
                ak = ak.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2) # B, #heads, T , Head Sizw
                av = av.view(1, aT, self.n_head, head_size).repeat(B, 1, 1, 1).transpose(1, 2)
                adapter_kv_cache = (ak, av)

            amask = torch.ones(q.shape[-2], ak.shape[-2], dtype=torch.bool, device=x.device)
            ay = F.scaled_dot_product_attention(q, ak, av, attn_mask=amask, dropout_p=0.0, is_causal=False)
            y = y + self.gating_factor * ay

            if w_kv_cache is not None:
                key, value = w_kv_cache
            else:

            # will insert model fusion here
            # convert shape from B 20 1500 64 to  B 20 1500 64
                key = self.whisper_key(audio_features)
                key = self.projection_rms_key(key) # put layernorm after so that the structure of audio features dont get ruined
                key = key @ self.projection_key_matrix_down
                key = F.silu(key)
                key = key @ self.projection_key_matrix_up
                key = key.view(B,20,1500,64)

                value = self.whisper_value(audio_features)
                value = self.projection_rms_value(value)
                value = value @ self.projection_key_matrix_down
                value = F.silu(value)
                value = value @ self.projection_key_matrix_up

                value = value.view(B,20,1500,64)

                #have to pad them to shape 
                padded_keys = torch.randn([B,32,1500,128],device=x.device, dtype=x.dtype)
                padded_values = torch.randn([B,32,1500,128],device=x.device, dtype=x.dtype)
            
                for num in range(B):
                    padded_keys[num] = torch.randn([1500,128]).repeat(32, 1, 1)
                    padded_keys[num,:20,:,:64] = key[num]

                    padded_values[num] = torch.randn([1500,128]).repeat(32, 1, 1)
                    padded_values[num,:20,:,:64] = value[num]
                
                key = padded_keys
                value = padded_values

                w_kv_cache = (key, value)

            q = q @ self.projection_query_matrix_128to128
            q = q.permute(0,2,3,1) @ self.projection_query_matrix_32to32
            q = q.permute(0,3,1,2)
        
            wmask = torch.ones(q.shape[-2], key.shape[-2], dtype=torch.bool, device=x.device)
            wy = F.scaled_dot_product_attention(q, key, value, attn_mask=wmask, dropout_p=0.0, is_causal=False)
            y = y + self.projection_gating_factor * wy


        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, kv_cache, adapter_kv_cache, w_kv_cache


class Block(nn.Module):
    """The implementation is identical to `lit_llama.model.Block` with the exception that
    we replace the attention layer where adaption is implemented."""

    def __init__(self, config: LLaMAConfig, block_idx: int) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, block_idx)
        self.rms_2 = RMSNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        audio_features: torch.Tensor,
        rope: RoPECache,
        mask: torch.Tensor,
        max_seq_length: int,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        adapter_kv_cache: Optional[KVCache] = None,
        w_kv_caches: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache], Optional[KVCache]]:
        h, new_kv_cache, new_adapter_kv_cache, new_w_kv_caches = self.attn(
            self.rms_1(x), audio_features, rope, mask, max_seq_length, input_pos, kv_cache, adapter_kv_cache, w_kv_caches
        )
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache, new_adapter_kv_cache, new_w_kv_caches


class LLaMA(llama.LLaMA):
    """The implementation is identical to `lit_llama.model.LLaMA` with the exception that
    the `Block` saves the layer index and passes it down to the attention layer."""

    def __init__(self, config: LLaMAConfig) -> None:
        nn.Module.__init__(self)
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        #self.padding = torch.nn.Parameter(torch.zeros([1500,32*128]))
        #self.padding_key = torch.nn.Parameter(   torch.zeros([1500,128]).fill_diagonal_(1).repeat(32, 1, 1)   )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                h=nn.ModuleList(Block(config, i) for i in range(config.n_layer)),
                ln_f=RMSNorm(config.n_embd),
            )
        )

        self.rope_cache: Optional[RoPECache] = None
        self.mask_cache: Optional[torch.Tensor] = None
        self.kv_caches: List[KVCache] = []
        self.adapter_kv_caches: List[KVCache] = []
        self.w_kv_caches: List[KVCache] = []


    @classmethod
    def from_name(cls, name: str):
        return cls(LLaMAConfig.from_name(name))

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        self.adapter_kv_caches.clear()
        self.w_kv_caches.clear()
        

    def forward(self, idx: torch.Tensor, audio_features: torch.Tensor, max_seq_length: Optional[int] = None, input_pos: Optional[torch.Tensor] = None,) -> Union[torch.Tensor, Tuple[torch.Tensor, List[KVCache]]]:
        B, T = idx.size()

        block_size = self.config.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert T <= max_seq_length, f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert max_seq_length <= block_size, f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert T <= block_size, f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx) # 2048,64,2
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx) # 1,1,2048,2048

        if input_pos is not None: # input pos is only passed durting inference ; hence this is a gate that opens during inference
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]


        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        # key = key.view(B,20,1500,64)
        # value = value.view(B,20,1500,64)

        # padded_keys = torch.zeros([B,32,1500,128],device=x.device, dtype=x.dtype)
        # padded_values = torch.zeros([B,32,1500,128],device=x.device, dtype=x.dtype)

        # for num in range(B):
        #     # padded_keys[num] = torch.cat( (  key[num,:,:] + self.padding[:,:1280]  , self.padding[:,1280:]   ),dim = -1  )
        #     # padded_values[num] = torch.cat( (  key[num,:,:] + self.padding[:,:1280]  , self.padding[:,1280:]   ),dim = -1  )
        #     padded_keys[num] = self.padding_key
        #     padded_keys[num,:20,:,:64] = key[num]

        #     padded_values[num] = self.padding_key
        #     padded_values[num,:20,:,:64] = value[num]

        if input_pos is None:  # proxy for use_cache=False aka proxy for training time 
            for block in self.transformer.h:
                x, *_ = block(x, audio_features ,rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.config.n_embd // self.config.n_head
                cache_shape = (B, self.config.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (torch.zeros(cache_shape, device=x.device, dtype=x.dtype), torch.zeros(cache_shape, device=x.device, dtype=x.dtype))
                    for _ in range(self.config.n_layer)
                ]
            if not self.adapter_kv_caches:
                self.adapter_kv_caches = [None for _ in range(self.config.n_layer)]
            if not self.w_kv_caches:
                self.w_kv_caches = [None for _ in range(self.config.n_layer)] # the first two layers will be empty because adapters start from 2

            for i, block in enumerate(self.transformer.h):
                # in Whisper, the audio features from the the encoder are constant (not updated be each block). We follow the same, Hence I am not getting the Key and Value as output
                x, self.kv_caches[i], self.adapter_kv_caches[i] , self.w_kv_caches[i] = block(
                    x,  audio_features ,rope, mask, max_seq_length, input_pos, self.kv_caches[i], self.adapter_kv_caches[i], self.w_kv_caches[i]
                )

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits


def mark_only_adapter_as_trainable(model: LLaMA) -> None:
    """Sets `requires_grad=False` for all non-adapter weights."""
    for name, param in model.named_parameters():
        param.requires_grad = "adapter_wte" in name or "gating_factor" in name or 'padding' in name or 'projection' in name 


def adapter_state_from_state_dict(state_dict: dict) -> dict:
    """Returns the model state dict with only the adapter weights for saving."""
    return {name: param for name, param in state_dict.items() if "adapter_wte" in name or "gating_factor" in name or 'padding' in name or 'projection' in name }