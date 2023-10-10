"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder, DTEncoder, DTDecoder
from utils import LossWithIntermediateLosses


@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)

        # quantized codebook dim为(vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 这里就是VQVAE的码本

        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(x, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None

        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        # z_q是VQVAE码本里面的embedding z
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()


        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        # 把[0, 1]转换到[-1, 1]
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        # 把[-1, 1]转换到[0, 1]
        return y.add(1).div(2)


class StateBasedTokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int,
                 encoder: DTEncoder, decoder: DTDecoder, with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        ### 按照上面的tokenizer应该要补一个conv1d
        self.pre_quant_conv = torch.nn.Conv1d(encoder.config.hidden_size, embed_dim, 1)

        # quantized codebook dim: (vocab_size, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        ### 按照上面的tokenizer应该要补一个conv1d
        self.post_quant_conv = torch.nn.Conv1d(embed_dim, decoder.config.hidden_size, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None


    def __repr__(self) -> str:
        return "statebasedtokenizer"
    
    def forward(self, obs: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        outputs = self.encode(obs, should_preprocess)
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions
    
    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t n o -> (b t) n o'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
        # - iris uses 1.0 by default
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()
        
        reconstruction_loss = torch.abs(observations - reconstructions)

        # 这里是对于像素设计的perceptual loss，我们如果使用state based env，这里是无法使用的
        # perceptual_loss = torch.mean(self.lpips(observations, reconstructions))
        
        ###TODO### 可以利用global state加入类似lpips的perceptual loss

        # 理论上VQ-VAE的loss只用包括reconstruction loss和commitment loss
        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss)

    def encode(self, obs: torch.Tensor, should_preprocess: bool = False):
        if should_preprocess:
            obs = self.preprocess_input(obs)        
        shape = obs.shape # (..., N, Obs_dim), N -> Nums of agents
        obs = obs.view(-1, *shape[-2:])
        z = self.encoder(obs)

        ##### Note: 这部分或许可以不用，因为encoder直接映射到了embedding层
        z = self.pre_quant_conv(z)
        ##### ---------------------------------------------------------
        
        b, n, e = z.shape
        z_flattened = rearrange(z, 'b n e -> (b n) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())
        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b n) e -> b n e', b=b, n=n, e=e).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-2], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-2], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-2], -1)

        return TokenizerEncoderOutput(z, z_q, tokens)
    
    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False):
        shape = z_q.shape # (..., N, embed_dim)
        z_q = z_q.view(-1, *shape[-2:])

        ##### Note: 这部分或许可以不用，因为encoder直接映射到了embedding层
        z_q = self.post_quant_conv(z_q)
        ##### ---------------------------------------------------------

        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-2], rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)

        return rec
        
    ###TODO###
    def preprocess_input(self, x: torch.Tensor):
        batch_size, seq_length = x.shape[0], x.shape[1]
        pass
    
    def postprocess_output(self, y: torch.Tensor):
        pass