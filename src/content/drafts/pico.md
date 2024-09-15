---
title: "Grouped Query Attention in torch"
description: "self-attention for the gpu-poor"
pubDate: "Jul 13 2024"
tags: ["python"]
---

I've been training my own llm from scratch (repo [here](https://github.com/nnethercott/picoGPT) if you're curious 😉), taking inspiration from projects like Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT), and [TinyLlama](https://github.com/jzhang38/TinyLlama). Along the way I've been making a few updates to reflect more recent architectural trends in the community and which make training less compute-intensive. I was going to be training on free Google colab instances which max out at 16GiB VRAM so this memory consideration was pretty critical for me.

A key design choice which helped me to this end was the [grouped query attention](https://arxiv.org/abs/2305.13245) block. GQA is a sparse version of the normally dense self attention mechanism. By changing the number of query groups you can interpolate between multi-query attention and SA. Of course using fewer parameters directly impacts the model's capacity for fitting the training data, but in practice GQA works quite well (plus we're on the trajectory to sub 1B parameter models anyways).

<!-- ``` -->
<!--    ┌───┐┌───┐┌───┐┌───┐      ┌───┐    ┌───┐              ┌───┐ -->
<!--    │ v ││ v ││ v ││ v │      │ v │    │ v │              │ v │ -->
<!--    └───┘└───┘└───┘└───┘      └───┘    └───┘              └───┘ -->
<!--      │    │    │    │          │        │                  │ -->
<!--    ┌───┐┌───┐┌───┐┌───┐      ┌───┐    ┌───┐              ┌───┐ -->
<!--    │ k ││ k ││ k ││ k │      │ k │    │ k │              │ k │ -->
<!--    └───┘└───┘└───┘└───┘      └───┘    └───┘              └───┘ -->
<!--      │    │    │    │       ┌──┴──┐  ┌──┴──┐       ┌────┬──┴─┬────┐ -->
<!--    ┌───┐┌───┐┌───┐┌───┐   ┌───┐┌───┐┌───┐┌───┐   ┌───┐┌───┐┌───┐┌───┐ -->
<!--    │ q ││ q ││ q ││ q │   │ q ││ q ││ q ││ q │   │ q ││ q ││ q ││ q │ -->
<!--    └───┘└───┘└───┘└───┘   └───┘└───┘└───┘└───┘   └───┘└───┘└───┘└───┘ -->
<!--    ◀──────────────────▶   ◀──────────────────▶   ◀──────────────────▶ -->
<!--             SA                    GQA                   MQA -->
<!--      n_query_groups=4       n_query_groups=2      n_query_groups=1 -->
<!---->
<!-- ``` -->

<div style="text-align: center;">
    <img src="https://klu.ai/_next/static/media/what-is-grouped-query-attention-gqa.32669ace.png" style="width: 100%; display: block; margin: 0 auto;">
</div>

Fun aside, you should read [this blog post](https://mobiusml.github.io/1bit_blog/) and [this paper](https://arxiv.org/abs/2402.17764) to understand what I meant by that comment above^ I'm thinking in the future of manually implementing low bit quantization in my custom and I'd be taking a bunch of inspiration from those papers.

When you get down to it I think most people with a basic understanding of linear algebra can piece together how SA works and is implemented, since we're just talking about a bunch of linear operations and reshapes. So with that being said i'll drop the code below !

<span style="font-size:0.85em;">

```python
import torch  
from torch import nn 
import torch.nn.functional as F

class CausalSelfAttention(nn.Module):
    """
    rotary positional embeddings from: https://github.com/lucidrains/rotary-embedding-torch
    Supports: GQA, MHA, SA based on choices for `n_query_groups` in config
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")

        shape = (
            config.n_head + 2 * config.n_query_groups
        ) * config.head_size  # n_head per query + (k+v)*n_query_groups

        self.c_attn = config.linear_cls(config.n_embd, shape, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.proj = config.linear_cls(
            config.head_size * config.n_head, config.n_embd, bias=config.bias
        )

        if not self.flash:
            print(
                "WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0"
            )

            # (1,1,bsz,bsz) to match (B,nh,T,hs) dims
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        self.rotary_emb = RotaryEmbedding(config.head_size, theta=config.rope_theta)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)

        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2  # queries per group + 1 key + 1 value
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )

        # q shape (B,T, n_query_groups, q_per_kv, head_size)
        # k and v shape (B,T, n_query_groups, 1, head_size)

        # splits total_qkv into amount per q, per k, per v
        q, k, v = qkv.split(
            (q_per_kv, 1, 1), dim=-2
        )  

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)

        y = self.scaled_dot_product_attention(q, k, v)
        y = y.reshape(B, T, self.config.n_head * self.config.head_size)

        y = self.proj(y)
        return y

    def scaled_dot_product_attention(self, q, k, v):
        T = q.shape[1]

        q = q.transpose(1, 2)  # (B,T,nh_q, hs) -> (B,nhs,T,hs)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # rotary positional embeddings from lucidrains/rotary-embedding-torch
        q = self.rotary_emb.rotate_queries_or_keys(q)
        k = self.rotary_emb.rotate_queries_or_keys(k)

        if q.shape != k.shape:
            # repeat k,v enough times so we can shove into F.scaled_dot_product_attention
            k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)

        # newer cuda gpus
        if self.flash:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att) if self.training else nn.Identity(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        return y.transpose(1, 2).contiguous()

```

</span>

add a table here with num params as a function of the query groups.
