# BiLSTM Language Model for sentence probability
import torch
import torch.nn as nn


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
        # First thing is to call the superclass initializer
        super().__init__()

        # Get vocab size and embedding dimension from the pretrained embeddings
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
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        ff_in = self.dropout(lstm_out)

        logits = self.model['lm_head'](ff_in)

        return logits  # (batch, seq_len, vocab_size)