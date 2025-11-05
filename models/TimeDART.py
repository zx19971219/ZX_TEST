import torch
import torch.nn as nn
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.TimeDART_EncDec import (
    ChannelIndependence,
    AddSosTokenAndDropLast,
    AddSosToken,
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
        self.block_size = args.block_size
        
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

        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model,
            dropout=self.dropout,
        )

        sos_token = torch.randn(1, args.block_size, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)

        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(
            sos_token=self.sos_token,
        )
        self.add_sos_token = AddSosToken(
            sos_token=self.sos_token,
        )

        # Encoder (Casual Trasnformer)
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            block_size=args.block_size,
            device=self.device,
            scheduler=args.scheduler,
        )
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            block_size=args.block_size,
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
            self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1
            self.context_len = self.seq_len
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                cond_dim=args.cond_dim,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                block_size=args.block_size,
                out_channels=args.patch_len,
                context_len=self.context_len,
                model_length=self.seq_len,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
            )

            self.projection = FlattenHead(
                seq_len=self.seq_len,
                d_model=self.patch_len,
                pred_len=args.input_len,
                dropout=args.head_dropout,
            )

        elif self.task_name == "finetune":
            self.seq_len = int((self.pred_len - self.patch_len) / self.stride) + 1
            self.context_len = int((self.input_len - self.patch_len) / self.stride) + 1
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                cond_dim=args.cond_dim,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                block_size=args.block_size,
                out_channels=args.patch_len,
                context_len=self.context_len,
                model_length=self.block_size,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
            )
            self.head = FlattenHead(
                seq_len=self.seq_len,
                d_model=args.patch_len,
                pred_len=args.pred_len,
                dropout=args.head_dropout,
            )

    def pretrain(self, x):
        # [batch_size, input_len, num_features]
        batch_size, input_len, num_features = x.size()
        if self.use_norm:
            # Instance Normalization
            means = torch.mean(
                x, dim=1, keepdim=True
            ).detach()  # [batch_size, 1, num_features], detach from gradient
            x = x - means  # [batch_size, input_len, num_features]
            stdevs = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()  # [batch_size, 1, num_features]
            x = x / stdevs  # [batch_size, input_len, num_features]

        # Channel Independence
        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        # Patch
        x_patch = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # For Casual Transformer
        x_embedding = self.enc_embedding(
            x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding = self.add_sos_token_and_drop_last(
            x_embedding
        )  # [batch_size * num_features, seq_len, d_model]
        x_embedding, _ = self.positional_encoding(x_embedding)
        x_out = self.encoder(
            x_embedding,
            is_mask=True,
        )  # [batch_size * num_features, seq_len, d_model]

        # Noising Diffusion
        noise_x_patch, noise, t = self.diffusion(
            x_patch
        )  # [batch_size * num_features, seq_len, patch_len]
        noise_x_embedding = self.enc_embedding(
            noise_x_patch
        )  # [batch_size * num_features, seq_len, d_model]
        noise_x_embedding, _ = self.positional_encoding(noise_x_embedding)

        # For Denoising Patch Decoder
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding,
            key=x_out,
            value=x_out,
            t=t,
            is_tgt_mask=True,
            is_src_mask=True,
        )  # [batch_size * num_features, seq_len, patch_len]

        # For Decoder
        predict_x = predict_x.reshape(
            batch_size, num_features, -1, self.patch_len
        )  # [batch_size, num_features, seq_len, patch_len]
        predict_x = self.projection(predict_x)  # [batch_size, input_len, num_features]

        # Instance Denormalization
        if self.use_norm:
            predict_x = predict_x * (stdevs[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]
            predict_x = predict_x + (means[:, 0, :].unsqueeze(1)).repeat(
                1, input_len, 1
            )  # [batch_size, input_len, num_features]

        return predict_x
    
    def sample(self, context, xt):
        # context: [batch_size * num_features, seq_len, patch_len]
        # xt: [batch_size * num_features, block_size, patch_len]
        total_series = torch.zeros(xt.shape[0], self.context_len + self.block_size, self.d_model).to(context.device)  # [batch_size * num_features, seq_len+block_size, d_model]
        _, total_pe = self.positional_encoding(total_series)  # [batch_size * num_features, seq_len+block_size, d_model]
        
        lookback_window = context[:, -self.context_len:, :]
        lookback_window_embedding = self.enc_embedding(lookback_window)  # [batch_size * num_features, seq_len, d_model]
        lookback_window_embedding = lookback_window_embedding + total_pe[:, -self.context_len:, :]
        
        lookback_window_embedding = self.encoder(
            lookback_window_embedding,
            is_mask=False,
        )  # [batch_size * num_features, seq_len+block_size, d_model]
        
        num_steps = 100
        timesteps = torch.linspace(self.time_steps - 1, 0, num_steps, dtype=torch.long).to(xt.device)
        for timestep in timesteps:
            xt_embedding = self.enc_embedding(xt)  # [batch_size * num_features, block_size, d_model]
            xt_embedding = xt_embedding + total_pe[:, -self.block_size:, :] # [batch_size * num_features, block_size, d_model]
            timestep = timestep.expand(xt.shape[0], self.block_size)
            
            pred_x0 = self.denoising_patch_decoder(
                query=xt_embedding,
                key=lookback_window_embedding,
                value=lookback_window_embedding,
                t=timestep,
                is_tgt_mask=False,
                is_src_mask=False,
                sample_mode=True,
            )
            xt = self.diffusion.p_sample(pred_x0, xt, timestep, clip_denoised=False)["sample"]
        return xt

    def forecast(self, x):
        batch_size, _, num_features = x.size()
        if self.use_norm:
            means = torch.mean(x, dim=1, keepdim=True).detach()
            x = x - means
            stdevs = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / stdevs

        x = self.channel_independence(x)  # [batch_size * num_features, input_len, 1]
        x = self.patch(x)  # [batch_size * num_features, seq_len, patch_len]

        # forecast
        num_strides = self.seq_len // self.block_size
        if self.seq_len % self.block_size != 0:
            raise ValueError("seq_len must be divisible by block_size")
        
        context = x  # [batch_size * num_features, seq_len, patch_len]
        for stride in range(num_strides):
            query = torch.randn(context.shape[0], self.block_size, self.patch_len, device=context.device)
            query = self.sample(context, query)
            context = torch.cat([context, query], dim=1)

        x = context[:, -self.seq_len:, :] # [batch_size * num_features, seq_len, patch_len]
        x = x.reshape(batch_size, num_features, -1, self.patch_len)  # [batch_size, num_features, seq_len, patch_len]
        x = self.head(x)  # [batch_size, pred_len, num_features]
        # denormalization
        if self.use_norm:
            x = x * (stdevs[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)
            x = x + (means[:, 0, :].unsqueeze(1)).repeat(1, self.pred_len, 1)

        return x
    
    def forward(self, batch_x):

        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        elif self.task_name == "finetune":
            dec_out = self.forecast(batch_x)
            return dec_out[:, -self.pred_len: , :]
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")


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