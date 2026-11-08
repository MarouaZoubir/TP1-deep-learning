
import torch
import torch.nn as nn
from torchvision import models
from torch.utils.tensorboard import SummaryWriter

# Imports du TP5
from model import ImageCaptioningModel
from dataset import train_loader, test_loader
from text_utils import (
    pretrained_embeddings,
    word2idx,
    idx2word,
    tokenize,
    untokenize
)


# ==========================================
# 1. INITIALISATION
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

# Chargement du ResNet50 pré-entraîné
cnn_model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Embeddings Word2Vec
embedding_model = pretrained_embeddings

# Dimension cachée du LSTM
lstm_hidden_dim = 512

# Initialisation du modèle
model = ImageCaptioningModel(
    cnn_model,
    embedding_model,
    lstm_hidden_dim
).to(device)

# Optimiseur
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# Fonction de perte
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

# TensorBoard
writer = SummaryWriter("runs/captioning")

num_epochs = 15


# ==========================================
# 2. BOUCLE D'ENTRAÎNEMENT
# ==========================================

for epoch in range(num_epochs):

    model.train()
    total_loss = 0

    for i, (images, captions) in enumerate(train_loader):

        optimizer.zero_grad(set_to_none=True)

        images = images.to(device)

        # Tokenisation du batch
        caps = torch.LongTensor(
            [tokenize(x, word2idx) for x in captions[0]]
        ).to(device)

        # Entrée : séquence sans le dernier mot
        outputs = model(images, caps[:, :-1])

        # Cible : séquence décalée d'un cran
        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            caps[:, 1:].reshape(-1)
        )

        # Backpropagation
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:

            writer.add_scalar(
                "Loss/Train_Step",
                total_loss / 10,
                epoch * len(train_loader) + i
            )

            total_loss = 0

    # ==========================================
    # 3. ÉVALUATION
    # ==========================================

    model.eval()
    val_loss = 0

    with torch.no_grad():

        for images, captions in test_loader:

            images = images.to(device)

            caps = torch.LongTensor(
                [tokenize(x, word2idx) for x in captions[0]]
            ).to(device)

            outputs = model(images, caps[:, :-1])

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                caps[:, 1:].reshape(-1)
            )

            val_loss += loss.item()

            # Visualisation d'une prédiction
            predictions = outputs[-1].argmax(dim=-1)

            pred_text = " ".join(
                untokenize(
                    predictions.cpu().numpy(),
                    idx2word
                )
            )

            true_text = " ".join(
                untokenize(
                    caps[-1][1:].cpu().numpy(),
                    idx2word
                )
            )

    # ==========================================
    # 4. AFFICHAGE ET LOGGING
    # ==========================================

    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    print(f"Train Loss: {total_loss:.4f}")
    print(f"Validation Loss: {val_loss / len(test_loader):.4f}")
    print(f"Prediction : {pred_text}")
    print(f"Target     : {true_text}")

    writer.add_scalar(
        "Loss/Validation",
        val_loss / len(test_loader),
        epoch
    )

    # Mise à jour du learning rate
    scheduler.step()

    writer.add_scalar(
        "LR",
        scheduler.get_last_lr()[0],
        epoch
    )

writer.close()

print("\nTraining completed!")

