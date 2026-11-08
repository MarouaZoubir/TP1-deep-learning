
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ==========================================
# 1. CNN ENCODER : RESNET50
# ==========================================

def create_cnn():
    """Create a pretrained ResNet50 spatial feature extractor."""

    resnet = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    # Keep convolutional layers, remove pooling and FC
    cnn = nn.Sequential(
        *list(resnet.children())[:-2]
    )

    # Freeze pretrained weights
    for param in cnn.parameters():
        param.requires_grad = False

    return cnn


# ==========================================
# 2. ATTENTION MODULE
# ==========================================

class AttentionModule(nn.Module):

    def __init__(self, feature_dim, hidden_dim):
        super().__init__()

        self.attention_layer = nn.Conv2d(
            feature_dim + hidden_dim,
            1,
            kernel_size=1
        )

    def forward(self, features, hidden_state):

        batch_size = features.size(0)
        height = features.size(2)
        width = features.size(3)

        # Expand hidden state over the spatial grid
        hidden_expanded = hidden_state.unsqueeze(2).unsqueeze(3)
        hidden_expanded = hidden_expanded.expand(
            -1, -1, height, width
        )

        # Concatenate visual features and hidden state
        concat_features = torch.cat(
            (features, hidden_expanded),
            dim=1
        )

        # Calculate attention scores
        scores = self.attention_layer(
            concat_features
        ).squeeze(1)

        # Softmax over all spatial regions
        attention_weights = F.softmax(
            scores.reshape(batch_size, -1),
            dim=1
        )

        attention_weights = attention_weights.view(
            batch_size, 1, height, width
        )

        # Weighted sum of spatial features
        context_vector = torch.sum(
            features * attention_weights,
            dim=(2, 3)
        )

        return context_vector, attention_weights


# ==========================================
# 3. CUSTOM LSTM WITH ATTENTION
# ==========================================

class LSTMWithAttention(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        attention_module,
        attention_size
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.attention_module = attention_module

        total_dim = input_size + hidden_size + attention_size

        self.lin_i = nn.Linear(total_dim, hidden_size)
        self.lin_f = nn.Linear(total_dim, hidden_size)
        self.lin_c = nn.Linear(total_dim, hidden_size)
        self.lin_o = nn.Linear(total_dim, hidden_size)

        self.init_weights()

    def init_weights(self):

        stdv = 1.0 / math.sqrt(self.hidden_size)

        for layer in [self.lin_i, self.lin_f, self.lin_c, self.lin_o]:
            layer.weight.data.uniform_(-stdv, stdv)
            layer.bias.data.zero_()

    def forward(self, x, features, init_states=None):

        batch_size, seq_size, _ = x.size()

        hidden_seq = []

        if init_states is None:

            h_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )

            c_t = torch.zeros(
                batch_size,
                self.hidden_size,
                device=x.device,
                dtype=x.dtype
            )

        else:
            h_t, c_t = init_states

        for t in range(seq_size):

            x_t = x[:, t, :]

            # Spatial attention
            context, _ = self.attention_module(
                features,
                h_t
            )

            # Concatenate context, input and hidden state
            concat = torch.cat(
                (context, x_t, h_t),
                dim=1
            )

            # LSTM gates
            i_t = torch.sigmoid(self.lin_i(concat))
            f_t = torch.sigmoid(self.lin_f(concat))
            g_t = torch.tanh(self.lin_c(concat))
            o_t = torch.sigmoid(self.lin_o(concat))

            # Update cell and hidden states
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(1))

        hidden_seq = torch.cat(hidden_seq, dim=1)

        return hidden_seq, (h_t, c_t)


# ==========================================
# 4. IMAGE CAPTIONING MODEL
# ==========================================
class ImageCaptioningModel(nn.Module):

    def __init__(self, cnn_model, embedding_model, lstm_hidden_dim):
        super().__init__()

        # CNN ResNet50 sans pooling ni couche fully connected
        self.cnn = nn.Sequential(
            *list(cnn_model.children())[:-2]
        )

        # Geler les paramètres du CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

        feature_dim = 2048

        # Embedding
        self.embeddings = embedding_model

        vocab_size = embedding_model.weight.shape[0]
        emb_dim = embedding_model.weight.shape[1]

        # Attention
        self.attention = AttentionModule(
            feature_dim=feature_dim,
            hidden_dim=lstm_hidden_dim
        )

        # LSTM personnalisé
        self.lstm = LSTMWithAttention(
            input_size=emb_dim,
            hidden_size=lstm_hidden_dim,
            attention_module=self.attention,
            attention_size=feature_dim
        )

        # Prédiction du prochain mot
        self.fc_out = nn.Linear(
            lstm_hidden_dim,
            vocab_size
        )

    def forward(self, images, captions):

        with torch.no_grad():
            cnn_features = self.cnn(images)

        embeddings = self.embeddings(captions)

        lstm_outputs, _ = self.lstm(
            embeddings,
            cnn_features
        )

        output = self.fc_out(lstm_outputs)

        return output
