import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    CausalTransformer,
    Diffusion,
    DenoisingPatchDecoder,
    DilatedConvEncoder,
    ClsEmbedding,
    ClsHead,
    OldClsHead,
    ClsFlattenHead,
    ARFlattenHead,
)
from layers.Embed import Patch, PatchEmbedding, PositionalEncoding, TokenEmbedding_TimeDART

class ContextHead(nn.Module):
    def __init__(self, d_model: int, cond_dim: int, seq_len: int, block_size: int):
        super(ContextHead, self).__init__()
        self.head1 = nn.Linear(d_model, cond_dim)
        self.head2 = nn.Linear(seq_len, block_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :return: [batch_size * num_features, block_size, cond_dim]
        """
        x = self.head1(x)  # (batch_size * num_features, seq_len, cond_dim)
        x = x.permute(0, 2, 1)  # (batch_size * num_features, cond_dim, seq_len)
        x = self.head2(x)  # (batch_size * num_features, cond_dim, block_size)
        x = x.permute(0, 2, 1)  # (batch_size * num_features, block_size, cond_dim)
        return x
    
    
class FlattenHead(nn.Module):
    def __init__(
        self,
        seq_len: int,
        d_model: int,
        pred_len: int,
        dropout: float,
    ):
        super(FlattenHead, self).__init__()
        self.pred_len = pred_len
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, num_features, seq_len, d_model]
        :return: [batch_size, pred_len, num_features]
        """
        x = self.flatten(x)  # (batch_size, num_features, seq_len * d_model)
        x = self.forecast_head(x)  # (batch_size, num_features, pred_len)
        x = self.dropout(x)  # (batch_size, num_features, pred_len)
        x = x.permute(0, 2, 1)  # (batch_size, pred_len, num_features)
        return x


class Model(nn.Module):
    """
    TimeDART
    """

    def __init__(self, args):
        super(Model, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.cond_dim = args.cond_dim
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.use_norm = args.use_norm
        self.channel_independence = ChannelIndependence()
        self.time_steps = args.time_steps
        self.ddim_method = args.ddim_method
        
        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.patch = Patch(
            patch_len=self.patch_len,
            stride=self.stride,
        )

        # Embedding
        self.enc_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            d_model=self.d_model,
        )

        self.seq_len = int((self.pred_len - self.patch_len) / self.stride) + 1
        self.context_len = int((self.input_len - self.patch_len) / self.stride) + 1
        self.sample_steps = args.sample_steps
        self.generate_len = args.generate_len
             
        if self.task_name == "pretrain_stage1":
            self.block_size = (self.seq_len // args.block_num)
            self.overlap_len = self.block_size
            
            self.projection = FlattenHead(
                seq_len=self.seq_len,
                d_model=args.patch_len,
                pred_len=self.pred_len,
                dropout=args.head_dropout,
            )
            
            self.diffusion = Diffusion(
                time_steps=args.time_steps,
                block_size=self.block_size,
                device=self.device,
                scheduler=args.scheduler,
            )
            
            self.context_map = ContextHead(
                d_model=self.d_model,
                cond_dim=self.cond_dim,
                seq_len=self.context_len,
                block_size=self.block_size,
            )

        else:
            self.block_size = int((self.generate_len - self.patch_len) / self.stride) + 1
            self.overlap_ratio = args.overlap_ratio
            self.overlap_len = int(self.block_size * (1 + self.overlap_ratio))
            # self.model_length = int((args.model_length - self.patch_len) / self.stride) + 1
            self.model_length = self.overlap_len
            
            self.diffusion = Diffusion(
                time_steps=args.time_steps,
                block_size=self.block_size,
                device=self.device,
                scheduler=args.scheduler,
            )
            
            self.context_map = ContextHead(
                d_model=self.d_model,
                cond_dim=self.cond_dim,
                seq_len=self.context_len,
                block_size=self.block_size,
            )
        
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            block_size=self.block_size,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )
        
        self.denoising_patch_decoder = DenoisingPatchDecoder(
            d_model=args.d_model,
            cond_dim=args.cond_dim,
            num_layers=args.d_layers,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            block_size=self.block_size,
            out_channels=args.patch_len,
            dropout=args.dropout,
            mask_ratio=args.mask_ratio,
        )
        
        self.head = FlattenHead(
            seq_len=self.block_size,
            d_model=args.patch_len,
            pred_len=self.generate_len,
            dropout=args.head_dropout,
        )
    
    def pretrain(self, x, y):
        batch_size, _, num_features = x.size()
        if self.use_norm:
            means_x = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means_x
            stdevs_x = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdevs_x
            
            means_y = torch.mean(y, dim=1, keepdim=True).detach()
            y = y - means_y
            stdevs_y = torch.sqrt(
            torch.var(y, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            y = y / stdevs_y

        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]
        y = self.channel_independence(y)  
        y = self.patch(y)  
        
        num_strides = self.seq_len // self.block_size
        x_embedding = self.enc_embedding(x)
        self.c_cond = self.context_map(x_embedding)
        self.c_cond = self.c_cond.repeat(1, num_strides, 1)
        
        yt, noise, t = self.diffusion(y)
        y_embedding = self.enc_embedding(y)
        yt_embedding = self.enc_embedding(yt)
        input_embedding = torch.cat([y_embedding, yt_embedding], dim=1)
            
        pred_y = self.denoising_patch_decoder(
            x=input_embedding,
            t=t,
            model_length=self.seq_len,
            context_cond=self.c_cond,
        )
            
        preds = pred_y.reshape(batch_size, num_features, -1, self.patch_len)  # [batch_size, num_features, seq_len, patch_len]
        # preds = self.projection(preds)
        preds = preds.chunk(num_strides, dim=2)  
        preds = torch.cat([self.head(block) for block in preds], dim=1)
                
        # denormalization
        if self.use_norm:
            preds = preds * (stdevs_x[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
            preds = preds + (means_x[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        return preds
    
    def pretrain_stage2(self, x, y):
        batch_size, _, num_features = x.size()
        if self.use_norm:
            means_x = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means_x
            stdevs_x = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdevs_x
            
            means_y = torch.mean(y, dim=1, keepdim=True).detach()
            y = y - means_y
            stdevs_y = torch.sqrt(
            torch.var(y, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            y = y / stdevs_y

        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]
        y = self.channel_independence(y)  
        y = self.patch(y)  
        
        x_embedding = self.enc_embedding(x)
        self.c_cond = self.context_map(x_embedding)
        
        num_strides = self.seq_len // self.block_size
        assert self.seq_len % self.block_size == 0, "seq_len must be divisible by block_size"
        
        context = x.clone()
        align_loss = []
        criterion = nn.MSELoss()
        
        for stride in range(num_strides):
            end_idx = (stride + 1) * self.block_size
            start_idx = max(end_idx - self.model_length, 0)
            fwd_idx = torch.arange(start_idx, end_idx)
            y_block = y[:, stride * self.block_size : (stride + 1) * self.block_size]
            
            yt, noise, t = self.diffusion(y_block)
            outputs = torch.cat([outputs, yt], dim=1) if stride != 0 else yt
            inputs = outputs[:, fwd_idx]
            
            inputs_embedding = self.enc_embedding(inputs) #[B, model_length, D]
            context_embedding = self.enc_embedding(context)
            # inputs_embedding = torch.cat([context_embedding, inputs_embedding], dim=1)
            
            pred_y = self.denoising_patch_decoder(
                x=inputs_embedding,
                t=t,
                model_length=end_idx-start_idx,
                context_cond=self.c_cond,
                sample=True,
            )
            
            outputs[:, fwd_idx] = pred_y
           
        pred_y = outputs.reshape(batch_size, num_features, -1, self.patch_len)  # [batch_size, num_features, pred_len, patch_len]
        pred_y = pred_y.chunk(num_strides, dim=2)  
        pred_y = torch.cat([self.head(block) for block in pred_y], dim=1)
                
        # denormalization
        if self.use_norm:
            pred_y = pred_y * (stdevs_x[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
            pred_y = pred_y + (means_x[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        # if align_loss:
        #     align_loss = torch.stack(align_loss).mean()
        # else:
        #     align_loss = torch.tensor(0.0, device=context.device)
        
        align_loss = torch.tensor(0.0, device=context.device)
        
        return pred_y, align_loss
    
    def sample(self, context, xt, stride=None):
        # context: [batch_size * num_features, context_len, patch_len]
        # xt: [batch_size * num_features, overlap_len or block_size, patch_len]
        context_embedding = self.enc_embedding(context) # [batch_size * num_features, context, d_model]
        
        timesteps = torch.linspace(
            start=0, 
            end=self.time_steps - 1, 
            steps=self.sample_steps, 
            dtype=torch.long,
            device=xt.device
        ).flip(0)
            
        for timestep in timesteps:
            # timestep = timestep.expand(xt.shape[0], self.block_size)
            # timestep_prefix = torch.zeros(xt.shape[0], xt.shape[1]-self.block_size, dtype=torch.long).to(xt.device)
            # timestep = torch.cat([timestep_prefix, timestep], dim=1)
            timestep_diffusion = timestep.expand(xt.shape[0], xt.shape[1])
            timestep = timestep.expand(xt.shape[0], self.block_size)
            # timestep_prefix = torch.zeros(xt.shape[0], xt.shape[1]-self.block_size, dtype=torch.long).to(xt.device)
            # timestep_diffusion = torch.cat([timestep_prefix, timestep], dim=1)
            
            input_embedding = self.enc_embedding(xt)  # [batch_size * num_features, overlap + block_size, d_model]
            # input_embedding = torch.cat([context_embedding, input_embedding], dim=1)
    
            pred_x0 = self.denoising_patch_decoder(
                x=input_embedding,
                t=timestep,
                model_length=xt.shape[1],
                context_cond=self.c_cond,
                sample=True,
            ) # [batch_size * num_features, overlap + block_size, patch_len]
            
            xt = self.diffusion.p_sample(
                pred_x0, 
                xt, 
                timestep_diffusion, 
                clip_denoised=False)["sample"]
            
        return xt

    def forecast(self, x):
        batch_size, _, num_features = x.size()
        if self.use_norm:
            means_x = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means_x
            stdevs_x = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdevs_x

        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # forecast
        num_strides = self.seq_len // self.block_size
        assert self.seq_len % self.block_size == 0, "seq_len must be divisible by block_size"
        
        x_embedding = self.enc_embedding(x)
        self.c_cond = self.context_map(x_embedding)
        
        context = x.clone()  # [batch_size * num_features, seq_len, patch_len]
        overlap = max(self.overlap_len - self.block_size, 0)
        has_overlap = overlap > 0
        for stride in range(num_strides):
            end_idx = (stride + 1) * self.block_size
            start_idx = max(end_idx - self.model_length, 0)
            fwd_idx = torch.arange(start_idx, end_idx)
            
            xt = torch.randn(context.shape[0], self.block_size, self.patch_len, device=context.device)
            outputs = torch.cat([outputs, xt], dim=1) if stride != 0 else xt
            inputs = outputs[:, fwd_idx]
            xt = self.sample(context, inputs, stride) # [batch_size * num_features, overlap_len or block_size, patch_len]
            
            outputs[:, fwd_idx] = xt

        # x = context[:, -self.seq_len:, :] # [batch_size * num_features, seq_len, patch_len]
        x = outputs.reshape(batch_size, num_features, -1, self.patch_len)  # [batch_size, num_features, seq_len, patch_len]
        x_blocks = x.chunk(num_strides, dim=2)  
        outputs = torch.cat([self.head(block) for block in x_blocks], dim=1)
                
        # denormalization
        if self.use_norm:
            outputs = outputs * (stdevs_x[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
            outputs = outputs + (means_x[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        return outputs
    
    def forward(self, batch_x, batch_y=None):

        if self.task_name == "pretrain_stage1":
            return self.pretrain(batch_x, batch_y)
        elif self.task_name == "pretrain_stage2":
            dec_out = self.forecast(batch_x)
            return dec_out[:, -self.pred_len: , :]
        else:
            dec_out = self.forecast(batch_x)
            return dec_out[:, -self.pred_len: , :]


class ClsModel(nn.Module):
    def __init__(self, args):
        super(ClsModel, self).__init__()
        self.input_len = args.input_len

        # For Model Hyperparameters
        self.d_model = args.d_model
        self.num_heads = args.n_heads
        self.feedforward_dim = args.d_ff
        self.dropout = args.dropout
        self.device = args.device
        self.task_name = args.task_name
        self.num_classes = args.num_classes

        # Patch
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 2
        padding = self.seq_len * self.stride - self.input_len

        # Embedding
        self.enc_embedding = ClsEmbedding(
            num_features=args.enc_in,
            d_model=args.d_model,
            kernel_size=args.patch_len,
            stride=args.stride,
            padding=padding,
        )

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )

        # Encoder (Casual Trasnformer)
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=self.device,
            scheduler=args.scheduler,
        )
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )
        # self.encoder = DilatedConvEncoder(
        #     in_channels=self.d_model,
        #     channels=[self.d_model] * args.e_layers,
        #     kernel_size=3,
        # )

        # Decoder
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
            )

            self.projection = ClsFlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=args.input_len,
                num_features=args.c_out,
                dropout=args.head_dropout,
            )

        elif self.task_name == "finetune":
            self.head = OldClsHead(
                seq_len=self.seq_len,
                d_model=args.d_model,
                num_classes=args.num_classes,
                dropout=args.head_dropout,
            )

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        # Instance Normalization
        # batch_size, input_len, num_features = x.size()
        # means = torch.mean(
        #     x, dim=1, keepdim=True
        # ).detach()  # [batch_size, 1, num_features], detach from gradient
        # x = x - means  # [batch_size, input_len, num_features]
        # stdevs = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        # ).detach()  # [batch_size, 1, num_features]
        # x = x / stdevs  # [batch_size, input_len, num_features]

        # For Casual Transformer
        x_embedding = self.enc_embedding(
            x
        )  # [batch_size, seq_len, d_model]
        x_embedding_bias = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size, seq_len, d_model]
        x_embedding_bias = self.positional_encoding(x_embedding_bias)
        x_out = self.encoder(
            x_embedding_bias,
            is_mask=True,
        )  # [batch_size, seq_len, d_model]

        # Noising Diffusion
        noise_x_patch, noise, t = self.diffusion(
            x
        )  # [batch_size, seq_len, patch_len]
        noise_x_embedding = self.enc_embedding(
            noise_x_patch
        )  # [batch_size, seq_len, d_model]
        noise_x_embedding = self.positional_encoding(noise_x_embedding)

        # For Denoising Patch Decoder
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding,
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=True,
        )  # [batch_size, seq_len, d_model]

        # For Decoder
        predict_x = self.projection(predict_x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        # predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
        #     1, input_len, 1
        # )  # [batch_size, input_len, num_features]
        # predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
        #     1, input_len, 1
        # )  # [batch_size, input_len, num_features]

        return predict_x

    def forecast(self, x):
        # batch_size, _, num_features = x.size()
        # means = torch.mean(x, dim=1, keepdim=True).detach()
        # x = x - means
        # stdevs = torch.sqrt(
        #     torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
        # ).detach()
        # x = x / stdevs

        x = self.enc_embedding(x)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)  # [batch_size, seq_len, d_model]
        x = self.encoder(
            x,
            is_mask=False,
        )  # [batch_size, seq_len, d_model]
        # forecast
        x = self.head(x)  # [bs, num_classes]

        return x
    
    def forward(self, batch_x):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            return self.forecast(batch_x)
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")