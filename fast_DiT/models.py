from typing import Any, Dict, Tuple, Type, Union, Optional
import torch
# from torch import Tensor, nn
from torch.nn import functional as F

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # FIXME: python 3.7

# import mindspore as ms
# from mindspore import Tensor, nn, ops
from torch import Tensor, nn
from torch.nn import GELU

# from flash_attention import MSFlashAttention

from dit import FinalLayer
# from utils import modulate

from torch.nn import LayerNorm

from torch.nn.init import xavier_uniform_, constant_, normal_

import math


__all__ = [
    "FiT",
    "FiT_models",
    "FiT_XL_2",
    "FiT_XL_4",
    "FiT_XL_8",
    "FiT_L_2",
    "FiT_L_4",
    "FiT_L_8",
    "FiT_B_2",
    "FiT_B_4",
    "FiT_B_8",
    "FiT_S_2",
    "FiT_S_4",
    "FiT_S_8",
]

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_channels=in_features, out_channels=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(in_channels=hidden_features, out_channels=out_features)
        self.drop = nn.Dropout(p=drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def complex_mult(x: Tensor, y: Tensor) -> Tensor:
    assert x.shape[-1] == y.shape[-1] == 2
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]

    # (a + ib)(c + id) = (ac - bd) + i(bc + ad)
    real_part = a * c - b * d
    imag_part = b * c + a * d
    return torch.stack([real_part, imag_part], dim=-1)



# TODO: Implement with torchtune.modules.RotaryPositionalEmbeddings
def apply_rotary_emb(q: Tensor, k: Tensor, freqs_cis: Tensor) -> Tuple[Tensor, Tensor]:
    q_shape = q.shape
    k_shape = k.shape
    # to complex
    q = q.reshape(q_shape[0], q_shape[1], q_shape[2], -1, 2)
    k = k.reshape(k_shape[0], k_shape[1], k_shape[2], -1, 2)  # b, h, n, d/2, 2
    freqs_cis = freqs_cis.reshape(
        freqs_cis.shape[0], 1, q_shape[2], -1, 2)  # b, 1, n, d/2, 2
    dtype = q.dtype
    q = complex_mult(q.to(torch.float32), freqs_cis).to(dtype)
    k = complex_mult(k.to(torch.float32), freqs_cis).to(dtype)
    # to real
    q = q.reshape(q_shape)
    k = k.reshape(k_shape)
    return q, k

class Attention(nn.Module):
    def __init__(self, dim_head: int, attn_drop: float = 0.0) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.attn_drop = nn.Dropout(p=attn_drop)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        sim = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            sim = sim.masked_fill(~mask[:, None, None, :], float('-inf'))
        attn = F.softmax(sim, dim=-1)
        attn = self.attn_drop(attn)
        return torch.matmul(attn, v)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        apply_rotate_embed: bool = False,
        enable_flash_attention: bool = False,
    ) -> None:
        super().__init__()

        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(p=proj_drop)
        self.attention = Attention(head_dim, attn_drop=attn_drop)

        self.apply_rotate_embed = apply_rotate_embed

        if enable_flash_attention:
            self.flash_attention = nn.MultiheadAttention(embed_dim=head_dim, num_heads=num_heads, dropout=attn_drop)
        else:
            self.flash_attention = None

    @staticmethod
    def _rearange_out(x: Tensor) -> Tensor:
        # (b, h, n, d) -> (b, n, h*d)
        b, _, n, _ = x.shape
        # Transpose the head and sequence length dimensions
        x = x.transpose(1, 2)
        x = x.reshape(b, n, -1)  # Flatten the last two dimensions
        return x

    def forward(self, x: Tensor, mask: Optional[Tensor] = None, freqs_cis: Optional[Tensor] = None) -> Tensor:
        h = self.num_heads
        B, N, _ = x.shape

        # (b, n, 3*h*d) -> (b, n, 3, h, d)  -> (3, b, h, n, d)
        qkv = self.qkv(x).reshape(B, N, 3, h, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.apply_rotate_embed:
            q, k = apply_rotary_emb(q, k, freqs_cis)

        if self.flash_attention and q.shape[2] % 16 == 0 and k.shape[2] % 16 == 0 and q.shape[-1] <= 256:
            # Using PyTorch logical and
            mask = mask[:, None, :] & mask[:, :, None]
            out = self.flash_attention(q, k, v, attn_mask=~mask)
        else:
            out = self.attention(q, k, v, mask=mask)

        # (b, h, n, d) -> (b, n, h*d)
        out = self._rearange_out(out)

        return self.proj_drop(self.proj(out))


class SwiGLU(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Module] = nn.SiLU,
        norm_layer: Optional[Type[nn.Module]] = None,
        # has_bias: bool = True,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1_g = nn.Linear(in_features, hidden_features)
        self.fc1_x = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop)
        self.norm = norm_layer(
            (hidden_features,)) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x: Tensor) -> Tensor:
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = self.act(x_gate) * x
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FiTBlock(nn.Module):
    """
    A FiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        ffn: Literal["swiglu", "mlp"] = "swiglu",
        pos: Literal["rotate", "absolute"] = "rotate",
        **block_kwargs: Any,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        apply_rotate_embed = pos == "rotate"
        self.attn = SelfAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, apply_rotate_embed=apply_rotate_embed, **block_kwargs
        )
        self.norm2 = LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        if ffn == "swiglu":
            mlp_hidden_dim = int(hidden_size * mlp_ratio *
                                 2 / 3)  # following LLaMA
            self.ffn = SwiGLU(
                in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.SiLU, drop=0
            )
        elif ffn == "mlp":
            mlp_hidden_dim = int(hidden_size * mlp_ratio)
            def approx_gelu(): return GELU(approximate="tanh")
            self.ffn = Mlp(in_features=hidden_size,
                           hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        else:
            raise ValueError(f"Unsupported ffn `{ffn}`")
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(
        self, x: Tensor, c: Tensor, mask: Optional[Tensor] = None, freqs_cis: Optional[Tensor] = None
    ) -> Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, axis=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa), mask=mask, freqs_cis=freqs_cis
        )
        x = x + \
            gate_mlp.unsqueeze(
                1) * self.ffn(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FiT(nn.Module):
    """
    FiT: Flexible Vision Transformer for Diffusion Model
    https://arxiv.org/abs/2402.12376

    Args:
        patch_size: patch size. Default: 2
        in_channels: The number of input channels in the input latent. Default: 4
        hidden_size: The hidden size of the Transformer model. Default: 1152
        depth: The number of blocks in this Transformer. Default: 28
        num_heads: The number of attention heads. Default: 16
        mlp_ratio: The expansion ratio for the hidden dimension in the MLP of the Transformer. Default: 4.0
        class_dropout_prob: The dropout probability for the class labels in the label embedder. Default: 0.1
        num_classes: The number of classes of the input labels. Default: 1000
        learn_sigma: Whether to learn the diffusion model's sigma parameter. Default: True
        ffn: Method to use in FFN block. Can choose SwiGLU or MLP. Default: swiglu
        pos: Method to use in positional encoding. Can choose absolute or rotate. Default: rotate
        block_kwargs: Additional keyword arguments for the Transformer blocks. for example, `{'enable_flash_attention':True}`. Default: {}
    """

    def __init__(
        self,
        patch_size: int = 2,
        in_channels: int = 4,
        hidden_size: int = 1152,
        depth: int = 28,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        class_dropout_prob: float = 0.1,
        num_classes: int = 1000,
        learn_sigma: bool = True,
        ffn: Literal["swiglu", "mlp"] = "swiglu",
        pos: Literal["rotate", "absolute"] = "rotate",
        block_kwargs: Dict[str, Any] = {},
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.pos = pos

        assert pos in ["absolute", "rotate"]
        assert ffn in ["swiglu", "mlp"]

        self.x_embedder = nn.Linear(
            self.in_channels * patch_size * patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(
            num_classes, hidden_size, class_dropout_prob)

        self.blocks = nn.Sequential(
            *[
                FiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                         ffn=ffn, pos=pos, **block_kwargs)
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(
            hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                xavier_uniform_(module.weight)
                if module.bias is not None:
                    constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize label embedding table:
        normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        normal_(self.t_embedder.mlp[0].weight, std=0.02)
        normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in FiT blocks:
        for block in self.blocks:
            constant_(block.adaLN_modulation[-1].weight, 0)
            constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        constant_(self.final_layer.linear.weight, 0)
        constant_(self.final_layer.linear.bias, 0)

    # def unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
    #     """
    #     x: (N, T, patch_size**2 * C)
    #     imgs: (N, C, H, W)
    #     """
    #     c = self.out_channels
    #     nh, nw = h // self.patch_size, w // self.patch_size
    #     x = x.reshape((x.shape[0], nh, nw, self.patch_size, self.patch_size, c))
    #     x = ops.transpose(x, (0, 5, 1, 3, 2, 4))
    #     imgs = x.reshape((x.shape[0], c, nh * self.patch_size, nw * self.patch_size))
    #     return imgs

    def unpatchify(self, x: Tensor, h: int, w: int) -> Tensor:
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        nh, nw = h // self.patch_size, w // self.patch_size
        x = x.reshape(
            (x.shape[0], nh, nw, self.patch_size, self.patch_size, c))
        x = x.permute(0, 5, 1, 3, 2, 4)  # Reorder dimensions
        imgs = x.reshape(
            (x.shape[0], c, nh * self.patch_size, nw * self.patch_size))
        return imgs

    # def patchify(self, x: Tensor) -> Tensor:
    #     N, C, H, W = x.shape
    #     nh, nw = H // self.patch_size, W // self.patch_size
    #     x = ops.reshape(x, (N, C, nh, self.patch_size, nw, self.patch_size))
    #     x = ops.transpose(x, (0, 2, 4, 3, 5, 1))  # N, nh, nw, patch, patch, C
    #     x = ops.reshape(x, (N, nh * nw, -1))
    #     return x

    def patchify(self, x: Tensor) -> Tensor:
        N, C, H, W = x.shape
        nh, nw = H // self.patch_size, W // self.patch_size
        x = x.reshape(N, C, nh, self.patch_size, nw, self.patch_size)
        x = x.permute(0, 2, 4, 3, 5, 1)  # N, nh, nw, patch, patch, C
        x = x.reshape(N, nh * nw, -1)
        return x

    def select_random_tokens( x: torch.Tensor, selection: int, mask: torch.Tensor) -> torch.Tensor:
        indices = torch.randperm(x.shape[1])

        x_shuffled = x[:, indices, :]
        mask_shuffled = mask[:, indices]

        mask_sorted = torch.argsort(mask_shuffled, dim=1, descending=True)

        x_sorted = x_shuffled[:, mask_sorted, :]

        return x_sorted[:selection]

    def forward(self, x: Tensor, t: Tensor, y: Tensor, pos: Tensor, mask: Tensor, h: int, w: int) -> Tensor:
        """
        Forward pass of FiT.
        x: (N, C, H, W) tensor of latent token
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        pos: (N, T, D) tensor of positional embedding or precomputed cosine and sine frequencies
        mask: (N, T) tensor of valid mask
        h: height of the input image latent
        w: width of the input image latent
        """
        # TODO: Check the shape of x and if pathify is necessary if already done in dataloader
        # _, _, h, w = x.shape
        # x = self.patchify(x)
        
        if self.pos == "absolute":
            # (N, T, D), where T = H * W / patch_size ** 2
            x = self.x_embedder(x) + pos.to(x.dtype)
        else:
            x = self.x_embedder(x)

        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)

        if self.pos == "rotate":
            freqs_cis = pos
        else:
            freqs_cis = None

        for block in self.blocks:
            x = block(x, c, mask=mask, freqs_cis=freqs_cis)  # (N, T, D)
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, h, w)  # (N, out_channels, H, W)
        return x

    # @ms.jit
    # def construct_with_cfg(
    #     self, x: Tensor, t: Tensor, y: Tensor, pos: Tensor, mask: Tensor, cfg_scale: Union[float, Tensor]
    # ) -> Tensor:
    #     """
    #     Forward pass of FiT, but also batches the unconditional forward pass for classifier-free guidance.
    #     """
    #     # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    #     half = x[: len(x) // 2]
    #     combined = ops.cat([half, half], axis=0)
    #     model_out = self.construct(combined, t, y, pos, mask)
    #     eps, rest = model_out[:, : self.in_channels], model_out[:, self.in_channels :]
    #     cond_eps, uncond_eps = ops.split(eps, len(eps) // 2, axis=0)
    #     half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    #     eps = ops.cat([half_eps, half_eps], axis=0)
    #     return ops.cat([eps, rest], axis=1)

    def construct_with_cfg(
        self, x: Tensor, t: Tensor, y: Tensor, pos: Tensor, mask: Tensor, cfg_scale: Union[float, Tensor]
    ) -> Tensor:
        """
        Forward pass of FiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y, pos, mask)
        eps, rest = model_out[:,
                              : self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


def FiT_XL_2(**kwargs):
    return FiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def FiT_XL_4(**kwargs):
    return FiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def FiT_XL_8(**kwargs):
    return FiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def FiT_L_2(**kwargs):
    return FiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def FiT_L_4(**kwargs):
    return FiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def FiT_L_8(**kwargs):
    return FiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def FiT_B_2(**kwargs):
    return FiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def FiT_B_4(**kwargs):
    return FiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def FiT_B_8(**kwargs):
    return FiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def FiT_S_2(**kwargs):
    return FiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def FiT_S_4(**kwargs):
    return FiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def FiT_S_8(**kwargs):
    return FiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


FiT_models = {
    "FiT-XL/2": FiT_XL_2,
    "FiT-XL/4": FiT_XL_4,
    "FiT-XL/8": FiT_XL_8,
    "FiT-L/2": FiT_L_2,
    "FiT-L/4": FiT_L_4,
    "FiT-L/8": FiT_L_8,
    "FiT-B/2": FiT_B_2,
    "FiT-B/4": FiT_B_4,
    "FiT-B/8": FiT_B_8,
    "FiT-S/2": FiT_S_2,
    "FiT-S/4": FiT_S_4,
    "FiT-S/8": FiT_S_8,
}
