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
                        FullAttention(False, itransformer_configs.factor, attention_dropout=itransformer_configs.dropout,
                                      output_attention=itransformer_configs.output_attention), itransformer_configs.d_model, itransformer_configs.n_heads),
                    itransformer_configs.d_model,
                    itransformer_configs.d_ff,
                    dropout=itransformer_configs.dropout,
                    activation=itransformer_configs.activation
                ) for _ in range(itransformer_configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(itransformer_configs.d_model)
        )
        self.projector = nn.Linear(itransformer_configs.d_model, itransformer_configs.pred_len, bias=True)

        """Hierarchical BERT head"""
        self.bert = BertModel(hbert_config)
        self.dropout = nn.Dropout(hbert_config.hidden_dropout_prob)
        self.classifier = nn.Linear(hbert_config.hidden_size, self.hbert_config.num_labels)

        """Extra mlp for the output of the heads"""
        # todo: add mlp instead of linear
        self.number_of_variate = 17

        self.linear = nn.Sequential(
            nn.Linear(self.number_of_variate, 1, bias=True),
            nn.Sigmoid()
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
        # TODO: delete decoder - read paper to understand
        dec_out = self.projector(x_numerical_encoded_output).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            # TODO: should it also stay? read paper to understand
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        # extra linear layer for the output

        dec_out = self.linear(dec_out)
        dec_out = dec_out.squeeze(2)

        """hierarchical bert head"""
        """
        :param x: A batch by sequence length integer tensor of token indices.
        :return: predicted log-probability vectors for each token based on the preceding tokens.
        """
        outputs = self.bert(attention_mask=None,
                            position_ids=None,
                            inputs_embeds=batch_x_textual)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]

        # todo: return concatenated output
        return dec_out
