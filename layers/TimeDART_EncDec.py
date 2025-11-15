import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.masking import generate_causal_mask, generate_self_only_mask, generate_partial_mask, generate_block_mask_train, generate_block_mask_sample, generate_block_mask_encoder
from utils.rotary_embed import Rotary, apply_rotary_pos_emb
import einops
from einops import rearrange

class ChannelIndependence(nn.Module):
    def __init__(
        self,
    ):
        super(ChannelIndependence, self).__init__()

    def forward(self, x):
        """
        :param x: [batch_size, input_len, num_features]
        :return: [batch_size * num_features, input_len, 1]
        """
        _, input_len, _ = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(-1, input_len, 1)
        return x


class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super(AddSosTokenAndDropLast, self).__init__()
        assert sos_token.dim() == 3
        self.sos_token = sos_token

    def forward(self, x):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        sos_token_expanded = self.sos_token.expand(
            x.size(0), -1, -1
        )  # [batch_size * num_features, block_size, d_model]
        x = torch.cat(
            [sos_token_expanded, x], dim=1
        )  # [batch_size * num_features, seq_len + block_size, d_model]
        x = x[:, :-self.sos_token.size(1), :]  # [batch_size * num_features, seq_len, d_model]
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.num_heads = num_heads
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def get_qkv(self, x, rotary_cos_sin=None):
        """
        input: x [B, S, D]
        output: qkv [B, S, 3, H, D/H]
        """
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
            
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3, h=self.num_heads, d=self.d_model // self.num_heads
        )
        
        if rotary_cos_sin is None:
            return qkv
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        return qkv
    
    def cross_attn(self, qkv, mask=None):
        scale = qkv.shape[-1]
        qkv = qkv.transpose(1, 3) # [b, s, 3, h, d] -> [b, h, 3, s, d]
        
        mask = mask.bool() if mask is not None else None
        x = F.scaled_dot_product_attention(
            query=qkv[:, :, 0],
            key=qkv[:, :, 1],
            value=qkv[:, :, 2],
            attn_mask=mask,
            is_causal=False,
            scale=1 / math.sqrt(scale))
        x = x.transpose(1, 2)
        x = rearrange(x, 'b s h d -> b s (h d)') # [b, s, h*d]
        return x

    def forward(self, x, rotary_cos_sin, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        qkv = self.get_qkv(x, rotary_cos_sin)
        attn_output = self.cross_attn(qkv, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        block_size: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.rotary_emb = Rotary(d_model // num_heads)
        self.block_size = block_size

    def forward(self, x, is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_block_mask_encoder(
                b=None, h=None, q_idx=torch.arange(seq_len)[:, None], 
                kv_idx=torch.arange(seq_len)[None, :], block_size=self.block_size).to(x.device) if is_mask else None
        rotary_cos_sin = self.rotary_emb(x)
        for layer in self.layers:
            x = layer(x, rotary_cos_sin, mask)

        x = self.norm(x)
        return x


class Diffusion(nn.Module):
    def __init__(
        self,
        time_steps: int,
        block_size: int,
        device: torch.device,
        scheduler: str = "cosine",
    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps

        if scheduler == "cosine":
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler=}")

        self.alpha = 1. - self.betas
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)
        self.gamma_prev = torch.cat([torch.tensor([1.0], device=self.device), self.gamma[:-1]])
        self.block_size = block_size
        
        self.sqrt_recip_gamma = torch.sqrt(1. / self.gamma)
        self.sqrt_recipm1_gamma = torch.sqrt(1. / self.gamma - 1)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1. - self.gamma_prev) / (1. - self.gamma)
        )
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.gamma_prev) / (1.0 - self.gamma)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.gamma_prev) * torch.sqrt(self.alpha) / (1.0 - self.gamma)
        )

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = (
            torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(beta_start, beta_end, self.time_steps)
        return betas
    
    def q_posterior_mean_variance(self, x0, xt, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x0.shape == xt.shape
        
        posterior_mean = self.posterior_mean_coef1[t].unsqueeze(-1) * x0 + self.posterior_mean_coef2[t].unsqueeze(-1) * xt
        posterior_variance = self.posterior_variance[t].unsqueeze(-1)
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t].unsqueeze(-1)
        
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == xt.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, pred_x0, xt, t, clip_denoised=True, denoised_fn=None):
        """
        compute the p(x_{t-1} | x_0, x_t, t)
        """
        if clip_denoised:
            pred_x0 = pred_x0.clamp(-1, 1)
        
        model_mean, posterior_variance, posterior_log_variance_clipped = self.q_posterior_mean_variance(x0=pred_x0, xt=xt, t=t)
        
        return {
            "mean": model_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance_clipped,
            "pred_x0": pred_x0,
        }
    
    def p_sample(self, pred_x0, xt, t, clip_denoised=True):
        """
        pred_x0: [batch_size * num_features, seq_len, patch_len]
        xt: [batch_size * num_features, seq_len, patch_len]
        t: [batch_size * num_features, seq_len]
        
        """
        out = self.p_mean_variance(pred_x0, xt, t, clip_denoised=clip_denoised)
        if (t==0).all():
            return {"sample": out["mean"], "pred_x0": out["pred_x0"]}
        
        noise = torch.randn_like(xt)
        sample = out["mean"] + torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x0": out["pred_x0"]}
    
    def predict_noise_from_x0(self, xt, t, x0):
        return (
                (self.sqrt_recip_gamma[t].unsqueeze(-1) * xt - x0) /
                self.sqrt_recipm1_gamma[t].unsqueeze(-1)
        )
    
    def ddim_sample(self, pred_x0, xt, t, eta=0):
        pred_noise = self.predict_noise_from_x0(xt, t, pred_x0)
        
        gamma_t = self.gamma[t].unsqueeze(-1)
        gamma_prev = self.gamma_prev[t].unsqueeze(-1)
        sigma_t = eta * ((1 - gamma_prev) / (1 - gamma_t) * (1 - gamma_t / gamma_prev)).sqrt()
        c = (1 - gamma_prev - sigma_t**2).sqrt()
        
        mean_pred = gamma_prev.sqrt() * pred_x0 + c * pred_noise
        if (t==0).all():
            return mean_pred
        noise = torch.randn_like(xt)
        x_prev = mean_pred + sigma_t * noise
        return x_prev

    def sample_time_steps(self, shape):
        # return torch.randint(0, self.time_steps, shape, device=self.device)
        block_size = self.block_size
        n = shape[-1]
        num_blocks = n // block_size
        if n % block_size != 0:
            raise ValueError(
            f"Input length {n} is not divisible by block_size {block_size}. "
            f"Each sample must contain an integer number of blocks."
        )
            
        _eps_b = torch.randint(0, self.time_steps, (shape[0], num_blocks), device=self.device) # [batch_size*feature_num, num_blocks]
        # expand to token level
        t = _eps_b
        if block_size != shape[-1]:
            t = t.repeat_interleave(block_size, dim=-1) # [batch_size*feature_num, input_len]
        return t

    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)  # [batch_size * num_features, seq_len, 1]
        # x_t = sqrt(gamma_t) * x + sqrt(1 - gamma_t) * noise
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x):
        # x: [batch_size * num_features, seq_len, patch_len]
        t = self.sample_time_steps(x.shape[:2])  # [batch_size * num_features, seq_len]
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t

class AdaLN(nn.Module):
    """
    Adaptive LayerNorm (AdaLN)
    """
    def __init__(self, d_model, cond_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 3 * d_model)  # scale, shift, gate
        )

    def forward(self, x, cond):
        # cond: [batch, d_model]
        scale, shift, gate = self.linear(cond).chunk(3, dim=-1)
        return self.norm(x) * (1 + scale) + shift, gate
    
class TimestepEmbedder(nn.Module):
  """
  Embeds scalar timesteps (can be [B, S]) into vector representations [B, S, hidden_size].
  """
  def __init__(self, hidden_size, frequency_embedding_size=256):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(frequency_embedding_size, hidden_size, bias=True),
      nn.SiLU(),
      nn.Linear(hidden_size, hidden_size, bias=True))
    self.frequency_embedding_size = frequency_embedding_size

  @staticmethod
  def timestep_embedding(t, dim, max_period=10000):
    """
        t: [B, S]
        return: [B, S, dim]
    """
    # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    half = dim // 2
    freqs = torch.exp(
      - math.log(max_period)
      * torch.arange(start=0, end=half).to(t.dtype).to(t.device)
      / half)
    args = t[..., None].float() * freqs[None, None, :]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
      embedding = torch.cat(
        [embedding,
         torch.zeros_like(embedding[..., :1])], dim=-1)
    return embedding

  def forward(self, t):
    t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
    t_emb = self.mlp(t_freq)
    return t_emb

class TransformerDecoderBlock(nn.Module):
    def __init__(
        self, d_model: int, model_length: int, cond_dim:int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerDecoderBlock, self).__init__()

        self.adaln1 = AdaLN(d_model, cond_dim)
        self.adaln2 = AdaLN(d_model, cond_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.dropout = nn.Dropout(dropout)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.model_length = model_length
        self.num_heads = num_heads
        self.d_model = d_model
        
    def get_qkv(self, x, rotary_cos_sin=None):
        """
        input: x [B, S, D]
        output: qkv [B, S, 3, H, D/H]
        """
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
            
        qkv = einops.rearrange(
            qkv,
            'b s (three h d) -> b s three h d',
            three=3, h=self.num_heads, d=self.d_model // self.num_heads
        )
        
        if rotary_cos_sin is None:
            return qkv
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype))
        return qkv
    
    def cross_attn(self, qkv, mask=None):
        scale = qkv.shape[-1]
        qkv = qkv.transpose(1, 3) # [b, s, 3, h, d] -> [b, h, 3, s, d]
        
        mask = mask.bool() if mask is not None else None
        x = F.scaled_dot_product_attention(
            query=qkv[:, :, 0],
            key=qkv[:, :, 1],
            value=qkv[:, :, 2],
            attn_mask=mask,
            is_causal=False,
            scale=1 / math.sqrt(scale))
        x = x.transpose(1, 2)
        x = rearrange(x, 'b s h d -> b s (h d)') # [b, s, h*d]
        return x

    def forward(self, x, cond, rotary_cos_sin=None, mask=None, sample=False):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param cond: [batch_size * num_features, seq_len, cond_dim]
        :param rotary_cos_sin: [1, seq_len, 3, 1, d_model]
        :param mask: [2 * seq_len, 2 * seq_len] 
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # cross-attention
        x_skip = x
        x, gate1 = self.adaln1(x, cond)
        if not sample:
            qkv_x0 = self.get_qkv(x[:, :self.model_length], rotary_cos_sin)
            qkv_xt = self.get_qkv(x[:, self.model_length:], rotary_cos_sin)
            qkv = torch.cat((qkv_x0, qkv_xt), dim=1)
        else: 
            qkv = self.get_qkv(x, rotary_cos_sin)
        attn_output = self.cross_attn(qkv, mask=mask)
        x = x_skip + self.dropout(attn_output) * gate1

        # Feed-forward network
        x_skip = x
        x, gate2 = self.adaln2(x, cond)
        ff_output = self.ff(x)
        x = x_skip + self.dropout(ff_output) * gate2

        return x

class FinalLayer(nn.Module):
  def __init__(self, d_model, out_channels, cond_dim):
    super().__init__()
    self.linear = nn.Linear(d_model, out_channels)
    self.linear.weight.data.zero_()
    self.linear.bias.data.zero_()
    self.adaln = AdaLN(d_model, cond_dim)

  def forward(self, x, cond):
    x, _ = self.adaln(x, cond)
    x = self.linear(x)
    return x

class DenoisingPatchDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        model_length: int,
        cond_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        block_size: int,
        out_channels: int,
        dropout: float,
        mask_ratio: float,
    ):
        super(DenoisingPatchDecoder, self).__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, model_length, cond_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.t_map = TimestepEmbedder(cond_dim)
        self.rotary_emb = Rotary(d_model // num_heads)
        self.final_layer = FinalLayer(d_model, out_channels, cond_dim)
        self.model_length = model_length

    def forward(self, x, t, context_cond=None, is_mask=True, sample=False):
        seq_len = x.size(1)
        cond = F.silu(self.t_map(t)) # [B, S] -> [B, S, cond_dim]
        if context_cond is not None:
            cond = cond + context_cond
        
        # pretraining
        if not sample:
            rotary_cos_sin = self.rotary_emb(x[:, :self.model_length])
            self.mask = generate_block_mask_train(
                b=None, h=None, q_idx=torch.arange(self.model_length*2)[:, None], 
                kv_idx=torch.arange(self.model_length*2)[None, :], block_size=self.block_size, n=self.model_length).to(x.device) if is_mask else None# [2*seq_len, 2*seq_len]
            cond_x0 = torch.zeros_like(cond).to(x.device)
            cond = torch.cat((cond_x0, cond), dim=1)
            
        # forecasting
        else:
            rotary_cos_sin = self.rotary_emb(x)
            self.mask = generate_block_mask_sample(
                b=None, h=None, q_idx=torch.arange(seq_len)[:, None], 
                kv_idx=torch.arange(seq_len)[None, :], block_size=self.block_size, n=seq_len-self.model_length).to(x.device) if is_mask else None# [seq_len, seq_len]
            cond_x0 = torch.zeros(cond.shape[0], seq_len-self.model_length, cond.shape[2]).to(x.device)
            cond = torch.cat((cond_x0, cond), dim=1)
        
        for layer in self.layers:
            x = layer(x, cond, rotary_cos_sin, mask=self.mask, sample=sample)
        
        x = self.final_layer(x, cond)
        x = x[:, -self.model_length:]
        return x


class SamePadConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(self.receptive_field - 1),  # 左填充
            dilation=dilation,
            groups=groups
        )
        
    def forward(self, x):
        out = self.conv(x)
        # 裁剪掉多余的未来时间步，确保与输入长度一致
        return out[:, :, :x.size(2)]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, final=False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or final else None
    
    def forward(self, x):
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual

class CausalTCN(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, kernel_size=3):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.depth = depth
        
        # First linear layer to map input_dims to hidden_dims
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        
        # Create a dilated causal convolutional encoder
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * (depth - 1) + [output_dims],
            kernel_size=kernel_size
        )
        
    def forward(self, x):
        # Input x is of shape [batch_size, seq_len, input_dims]
        
        # Flatten input (batch_size, seq_len, input_dims) -> (batch_size, seq_len, hidden_dims)
        x = self.input_fc(x)
        
        # Transpose for the convolution (batch_size, seq_len, hidden_dims) -> (batch_size, hidden_dims, seq_len)
        x = x.transpose(1, 2)
        
        # Apply dilated convolutions
        x = self.feature_extractor(x)  # [batch_size, hidden_dims, seq_len] -> [batch_size, output_dims, seq_len]
        
        # Transpose back to [batch_size, seq_len, output_dims]
        x = x.transpose(1, 2)
        
        return x


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):   
        """
        :param x: [batch_size, seq_len, input_dims]
        :return: [batch_size, seq_len, output_dims]
        """
        x = x.transpose(1, 2)
        return self.net(x).transpose(1, 2)


class ClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(ClsHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(seq_len * d_model, num_classes)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.dropout(x)
        return self.fc(x)


class OldClsHead(nn.Module):
    def __init__(self, seq_len, d_model, num_classes, dropout):
        super(OldClsHead, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)
    
    def forward(self, x):
        x = self.dropout(x)
        return self.fc(torch.max(x, dim=1)[0])


class ClsEmbedding(nn.Module):
    def __init__(self, num_features, d_model, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=num_features, 
            out_channels=d_model, 
            kernel_size=kernel_size, 
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.conv(x).transpose(1, 2) 


class ClsFlattenHead(nn.Module):
    def __init__(self, seq_len, d_model, pred_len, num_features, dropout):
        super(ClsFlattenHead, self).__init__()
        self.pred_len = pred_len
        self.num_features = num_features
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len * num_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # [batch_size, seq_len * d_model]
        x = self.dropout(x)  # [batch_size, seq_len * d_model]
        x = self.forecast_head(x)  # [batch_size, pred_len * num_features]
        return x.reshape(x.size(0), self.pred_len, self.num_features)


class ARFlattenHead(nn.Module):
    def __init__(
        self,
        d_model: int,
        patch_len: int,
        dropout: float,
    ):
        super(ARFlattenHead, self).__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(d_model, patch_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, seq_len * patch_len, num_features]
        """
        x = self.forecast_head(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.dropout(x)  # (batch_size, num_features, seq_len, patch_len)
        x = self.flatten(x)  # (batch_size, num_features, seq_len * patch_len)
        x = x.permute(0, 2, 1)  # (batch_size, seq_len * patch_len, num_features)
        return x