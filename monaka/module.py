# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from typing import Dict
from registrable import Registrable
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizerFast
from transformers import AutoModel, AutoConfig, T5EncoderModel


class LMEmbedding(nn.Module, Registrable):
   
   def __init__(self, *args, **kwargs) -> None:
    self.n_out = kwargs.get("n_out", 0)
    nn.Module.__init__(self)
    Registrable.__init__(self)

    @classmethod
    def from_config(cls, config: Dict):
        return cls(**config)
   

@LMEmbedding.register("AutoLM")
class AutoLMEmebedding(LMEmbedding):
    """
   TransformersのAutoConfigとAutoModelを利用するEmbedding

    Args:
        model (str):
            モデル名
        requires_grad (bool):
            LMを学習するかどうか
        use_scalar_mix (bool):
            ScalarMixを使うかどうか。使わない場合は最終レイヤ
        sclar_mix_dropout (float):
            ScalarMixのdropout
        use_attentions (bool):
            attentionを出力するかどうか
   """
   
    def __init__(self, model: str, requires_grad: bool, use_scalar_mix: bool, sclar_mix_dropout:float = 0.1, use_attentions: bool=False,
                max_length: int=512) -> None:
        self.model = model
        self.requires_grad = requires_grad
        self.use_scalar_mix = use_scalar_mix
        self.sclar_mix_dropout = use_scalar_mix
        self.use_attentions = use_attentions
        self.max_length = max_length

        self.config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                                    output_attentions=use_attentions)
            
        super().__init__(n_out=self.config.hidden_size)
        self.lm = AutoModel.from_pretrained(model, config=self.config)
        self.lm.requires_grad_(requires_grad)
        self.n_layers = self.config.num_hidden_layers
        self.pad_index = self.config.pad_token_id
        self.sclar_mix = ScalarMix(self.n_layers, sclar_mix_dropout)
        
    def __repr__(self):
        s = f"{self.model}, n_out={self.n_out}"
        s += f", pad_index={self.pad_index}"
        s += f", use_scalar_mix={self.use_scalar_mix}"
        if self.sclar_mix_dropout > 0:
            s += f", mix_dropout={self.sclar_mix_dropout}"
        if self.use_attentions:
            s += f", use_attentions={self.use_attentions}"
        if self.requires_grad:
            s += f", requires_grad={self.requires_grad}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, subwords):
        r"""
        Args:
            subwords (~torch.Tensor): ``[batch_size, subwordlen]``.
        Returns:
            ~torch.Tensor:
                BERT embeddings of shape ``[batch_size, seq_len, n_out]``.
        """

        mask = subwords.ne(self.pad_index)
        if not self.requires_grad:
            self.lm.eval()    # CHECK_ME (supar does not do)
        # [batch_size, n_subwords]
        # Outputs from the transformer:
        # - last_hidden_state: [batch, seq_len, hidden_size]
        # - pooler_output: [batch, hidden_size],
        # - hidden_states (optional): [[batch_size, seq_length, hidden_size]] * (1 + layers)
        # - attentions (optional): [[batch_size, num_heads, seq_length, seq_length]] * layers
        # print('<BERT, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        outputs = self.lm(subwords, attention_mask=mask.float())
        # print('BERT>, GPU MiB:', memory_allocated() // (1024*1024)) # DEBUG
        if self.use_scalar_mix:
            bert_idx = -2 if self.use_attentions else -1
            bert = outputs[bert_idx]
            # [n_layers, batch_size, n_subwords, hidden_size]
            bert = bert[-self.n_layers:]
            # [batch_size, n_subwords, hidden_size]
            bert = self.scalar_mix(bert)
        else:
            bert = outputs[0]
        

        return bert

@LMEmbedding.register("T5Encoder")
class T5EncoderEmbedding(AutoLMEmebedding):
   """
   T5Encoderを利用するEmbedding

    Args:
        model (str):
            モデル名
        requires_grad (bool):
            LMを学習するかどうか
        use_scalar_mix (bool):
            ScalarMixを使うかどうか。使わない場合は最終レイヤ
        sclar_mix_dropout (float):
            ScalarMixのdropout
        use_attentions (bool):
            attentionを出力するかどうか
        max_len (int):
            最大subword長 default 5120
   """
   
   def __init__(self, model: str, requires_grad: bool, use_scalar_mix: bool, sclar_mix_dropout:float = 0.1, use_attentions: bool=False, max_len: int=5120) -> None:
    self.model = model
    self.requires_grad = requires_grad
    self.use_scalar_mix = use_scalar_mix
    self.sclar_mix_dropout = sclar_mix_dropout
    self.use_attentions = use_attentions
    self.max_len = max_len

    self.config = AutoConfig.from_pretrained(model, output_hidden_states=True,
                                            output_attentions=use_attentions)
    self.lm = T5EncoderModel.from_pretrained(model, config=self.config)
    self.lm.requires_grad_(requires_grad)
    self.n_layers = self.config.num_hidden_layers
    self.pad_index = self.config.pad_token_id
    self.sclar_mix = ScalarMix(self.n_layers, sclar_mix_dropout)

    LMEmbedding.__init__(self, n_out=self.config.hidden_size)
    


class MLP(nn.Module):
    r"""
    Applies a linear transformation together with :class:`~torch.nn.LeakyReLU` activation to the incoming tensor:
    :math:`y = \mathrm{LeakyReLU}(x A^T + b)`

    Args:
        n_in (~torch.Tensor):
            The size of each input feature.
        n_out (~torch.Tensor):
            The size of each output feature.
        dropout (float):
            If non-zero, introduce a :class:`SharedDropout` layer on the output with this dropout ratio. Default: 0.
    """

    def __init__(self, n_in, n_out, dropout=0):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        r"""
        Args:
            x (~torch.Tensor):
                The size of each input feature is `n_in`.

        Returns:
            A tensor with the size of each output feature `n_out`.
        """

        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class ScalarMix(nn.Module):
    r"""
    Computes a parameterised scalar mixture of :math:`N` tensors, :math:`mixture = \gamma * \sum_{k}(s_k * tensor_k)`
    where :math:`s = \mathrm{softmax}(w)`, with :math:`w` and :math:`\gamma` scalar parameters.

    Args:
        n_layers (int):
            The number of layers to be mixed, i.e., :math:`N`.
        dropout (float):
            The dropout ratio of the layer weights.
            If dropout > 0, then for each scalar weight, adjust its softmax weight mass to 0
            with the dropout probability (i.e., setting the unnormalized weight to -inf).
            This effectively redistributes the dropped probability mass to all other weights.
            Default: 0.
    """

    def __init__(self, n_layers: int, dropout: float = 0.0):
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.gamma = nn.Parameter(torch.tensor([1.0]))
        self.dropout = nn.Dropout(dropout)

    def __repr__(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, tensors):
        r"""
        Args:
            tensors (list[~torch.Tensor]):
                :math:`N` tensors to be mixed.

        Returns:
            The mixture of :math:`N` tensors.
        """

        normed_weights = self.dropout(self.weights.softmax(-1))
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.gamma * weighted_sum
