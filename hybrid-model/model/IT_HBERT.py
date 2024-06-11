import torch
import torch.nn as nn

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
from layers.bert_layers import *
from model.BertModel import BertModel


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, itransformer_configs, hbert_config):
        super(Model, self).__init__()
        self.seq_len = itransformer_configs.seq_len
        self.pred_len = itransformer_configs.pred_len
        self.output_attention = itransformer_configs.output_attention
        self.use_norm = itransformer_configs.use_norm

        self.hbert_config = hbert_config

        """iTransformer head (Encoder-only architecture)"""
        # Embedding
        self.iTransformer_encoder_embedding = DataEmbedding_inverted(itransformer_configs.seq_len, itransformer_configs.d_model, itransformer_configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            itransformer_configs.factor,
                            attention_dropout=itransformer_configs.dropout,
                            output_attention=itransformer_configs.output_attention),
                        itransformer_configs.d_model,
                        itransformer_configs.n_heads),
                    itransformer_configs.d_model,
                    itransformer_configs.d_ff,
                    dropout=itransformer_configs.dropout,
                    activation=itransformer_configs.activation
                ) for _ in range(itransformer_configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(itransformer_configs.d_model)
        )

        """Hierarchical BERT head"""
        self.bert = BertModel(hbert_config)
        self.dropout = nn.Dropout(hbert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(hbert_config.hidden_size, self.hbert_config.num_labels)

        """Extra mlp for the output of the heads"""
        self.number_of_variate = 18

        flatten_input_size = self.number_of_variate * itransformer_configs.d_model

        self.mlp = nn.Sequential(
            nn.Linear(flatten_input_size, self.number_of_variate, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.number_of_variate, 2)
        )



    def forward(self, x_numerical_encoded, batch_x_textual):
        """iTransformer head"""
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_numerical_encoded.mean(1, keepdim=True).detach()
            x_numerical_encoded = x_numerical_encoded - means
            stdev = torch.sqrt(torch.var(x_numerical_encoded, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_numerical_encoded /= stdev

        _, _, N = x_numerical_encoded.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        x_numerical_encoded_output = self.iTransformer_encoder_embedding(x_numerical_encoded)

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        x_numerical_encoded_output, attns = self.encoder(x_numerical_encoded_output, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = x_numerical_encoded_output.permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        dec_out = dec_out.permute(0, 2, 1)

        """hierarchical bert head"""
        hbert_outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=batch_x_textual)

        hbert_pooled_output = hbert_outputs[1]

        """Output MLP head for the heads"""
        # add the output of the transformer to the output of the bert through second axis
        output = torch.cat((dec_out, hbert_pooled_output.unsqueeze(dim=1)), 1)

        # flatten the output
        output = output.view(output.size(0), -1)
        output = self.mlp(output)

        return output
