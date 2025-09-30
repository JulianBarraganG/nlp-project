# BiLSTM Language Model for sentence probability
import torch
import torch.nn as nn
import math


class BiLSTMLanguageModel(nn.Module):
    def __init__(self, 
                 pretrained_embeddings: torch.tensor, #type: ignore
                 lstm_dim: int, 
                 dropout_prob: float = 0.1
    ):
        """
        Initializer for basic BiLSTM network
        :param pretrained_embeddings: A tensor containing the pretrained BPE embeddings
        :param lstm_dim: The dimensionality of the BiLSTM network
        :param dropout_prob: Dropout probability
        """
        super().__init__()

        vocab_size = pretrained_embeddings.shape[0] # Size of the vocabulary
        embed_dim = pretrained_embeddings.shape[1] # Dimensionality of the embeddings

        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, a 2 layer BiLSTM, and a feed-forward output layer
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=vocab_size - 1),
            'bilstm': nn.LSTM(embed_dim, lstm_dim, 1, batch_first=True, dropout=dropout_prob, bidirectional=True),
            'lm_head': nn.Linear(2 * lstm_dim, vocab_size)
        })
        self.n_classes = vocab_size
        self.dropout = nn.Dropout(p=dropout_prob)
        
        # Initialize the weights of the model
        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['bilstm'].named_parameters()) + \
                     list(self.model['lm_head'].named_parameters())
        for n, p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :return: logits
        """
        
        # Get embeddings (b x sl x edim)
        embeds = self.model['embeddings'](inputs)

        # Pack padded: This is necessary for padded batches input to an RNN
        lstm_in = nn.utils.rnn.pack_padded_sequence(
            embeds, 
            input_lens.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )

        # Pass the packed sequence through the BiLSTM
        lstm_out, _ = self.model['bilstm'](lstm_in)

        # Unpack the packed sequence
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=inputs.size(1)
        )

        ff_in = self.dropout(lstm_out)

        logits = self.model['lm_head'](ff_in)

        return logits  # (batch, seq_len, vocab_size)
    

class BiLSTMClassifierModel(nn.Module):
    """
    Basic BiLSTM network
    """
    def __init__(
            self,
            pretrained_embeddings: torch.tensor, #type: ignore
            lstm_dim: int,
            n_classes: int,
            dropout_prob: float = 0.1,
    ):
        """
        Initializer for basic BiLSTM network
        :param pretrained_embeddings: A tensor containing the pretrained BPE embeddings
        :param lstm_dim: The dimensionality of the BiLSTM network
        :param dropout_prob: Dropout probability
        :param n_classes: The number of output classes
        """

        super().__init__()

        # We'll define the network in a ModuleDict, which makes organizing the model a bit nicer
        # The components are an embedding layer, a 2 layer BiLSTM, and a feed-forward output layer
        self.model = nn.ModuleDict({
            'embeddings': nn.Embedding.from_pretrained(pretrained_embeddings, padding_idx=pretrained_embeddings.shape[0] - 1),
            'bilstm': nn.LSTM(
                pretrained_embeddings.shape[1],
                lstm_dim,
                1,
                batch_first=True,
                dropout=dropout_prob,
                bidirectional=True),
            'cls': nn.Linear(2*lstm_dim, n_classes)
        })
        self.n_classes = n_classes
        self.dropout = nn.Dropout(p=dropout_prob)

        # Initialize the weights of the model
        self._init_weights()

    def _init_weights(self):
        all_params = list(self.model['bilstm'].named_parameters()) + \
                     list(self.model['cls'].named_parameters())
        for n,p in all_params:
            if 'weight' in n:
                nn.init.xavier_normal_(p)
            elif 'bias' in n:
                nn.init.zeros_(p)

    def forward(self, inputs, input_lens):
        """
        Defines how tensors flow through the model
        :param inputs: (b x sl) The IDs into the vocabulary of the input samples
        :param input_lens: (b) The length of each input sequence
        :return: (logits,)
        """

        # Get embeddings (b x sl x edim)
        embeds = self.model['embeddings'](inputs)

        # Pack padded: This is necessary for padded batches input to an RNN
        lstm_in = nn.utils.rnn.pack_padded_sequence(
            embeds,
            input_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        # Pass the packed sequence through the BiLSTM
        lstm_out, hidden = self.model['bilstm'](lstm_in)

        # Unpack the packed sequence --> (b x sl x 2*lstm_dim)
        lstm_out,_ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Max pool along the last dimension
        ff_in = self.dropout(torch.max(lstm_out, 1)[0])
        # Some magic to get the last output of the BiLSTM for classification (b x 2*lstm_dim)
        #ff_in = lstm_out.gather(1, input_lens.view(-1,1,1).expand(lstm_out.size(0), 1, lstm_out.size(2)) - 1).squeeze()

        # Get logits (b x n_classes)
        logits = self.model['cls'](ff_in).view(-1, self.n_classes)

        return logits

class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerClassifierModel(nn.Module):
    """
    Simple Transformer-based classifier for binary classification
    """
    def __init__(
            self,
            pretrained_embeddings: torch.tensor,
            d_model: int = 256,
            n_heads: int = 8,
            n_layers: int = 3,
            dim_feedforward: int = 512,
            n_classes: int = 2,
            dropout_prob: float = 0.1,
            max_len: int = 512
    ):
        """
        Initializer for Transformer classifier
        :param pretrained_embeddings: A tensor containing the pretrained BPE embeddings
        :param d_model: The dimensionality of the transformer model
        :param n_heads: Number of attention heads
        :param n_layers: Number of transformer layers
        :param dim_feedforward: Dimension of the feedforward network
        :param n_classes: The number of output classes
        :param dropout_prob: Dropout probability
        :param max_len: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_classes = n_classes
        
        # Embedding layer
        self.embeddings = nn.Embedding.from_pretrained(
            pretrained_embeddings, 
            padding_idx=pretrained_embeddings.shape[0] - 1
        )
        
        # Project embeddings to d_model if necessary
        embed_dim = pretrained_embeddings.shape[1]
        if embed_dim != d_model:
            self.embed_projection = nn.Linear(embed_dim, d_model)
        else:
            self.embed_projection = None
            
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Classification head
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(d_model, n_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize transformer weights"""
        for name, param in self.named_parameters():
            if 'embeddings' in name:
                continue  # Skip pretrained embeddings
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    def create_padding_mask(self, inputs, input_lens):
        """
        Create padding mask for attention
        :param inputs: (batch_size, seq_len) input token ids
        :param input_lens: (batch_size,) actual lengths of sequences
        :return: (batch_size, seq_len) boolean mask where True indicates padding
        """
        batch_size, seq_len = inputs.shape
        mask = torch.arange(seq_len, device=inputs.device).unsqueeze(0) >= input_lens.unsqueeze(1)
        return mask
        
    def forward(self, inputs, input_lens):
        """
        Forward pass through the transformer
        :param inputs: (batch_size, seq_len) The IDs into the vocabulary
        :param input_lens: (batch_size,) The length of each input sequence
        :return: logits (batch_size, n_classes)
        """
        batch_size, seq_len = inputs.shape
        
        # Get embeddings
        embeds = self.embeddings(inputs)  # (batch_size, seq_len, embed_dim)
        
        # Project to d_model if necessary
        if self.embed_projection is not None:
            embeds = self.embed_projection(embeds)
        
        # Scale embeddings (as in original transformer paper)
        embeds = embeds * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeds = embeds.transpose(0, 1)  # (seq_len, batch_size, d_model)
        embeds = self.pos_encoder(embeds)
        embeds = embeds.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Create padding mask
        padding_mask = self.create_padding_mask(inputs, input_lens)
        
        # Pass through transformer
        transformer_out = self.transformer(embeds, src_key_padding_mask=padding_mask)
        # (batch_size, seq_len, d_model)
        
        # Global max pooling over sequence dimension (ignoring padding)
        # Set padded positions to very negative values before max pooling
        mask_expanded = padding_mask.unsqueeze(-1).expand_as(transformer_out)
        transformer_out = transformer_out.masked_fill(mask_expanded, float('-inf'))
        pooled_output = torch.max(transformer_out, dim=1)[0]  # (batch_size, d_model)
        
        # Alternative: Use mean pooling instead of max pooling
        # transformer_out = transformer_out.masked_fill(mask_expanded, 0.0)
        # seq_lengths = input_lens.unsqueeze(1).float()
        # pooled_output = torch.sum(transformer_out, dim=1) / seq_lengths
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)  # (batch_size, n_classes)
        
        return logits