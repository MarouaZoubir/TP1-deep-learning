import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

from dataset import CardioDataset


# ============================================================
# Dataset et DataLoaders
# ============================================================

dataset = CardioDataset("data/cardio_train.csv")

generator = torch.Generator().manual_seed(42)

train_set, val_set, test_set = random_split(
    dataset,
    [0.8, 0.1, 0.1],
    generator=generator
)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=64,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=64,
    shuffle=False
)


# ============================================================
# MLP
# ============================================================

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Device
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Entraînement
# ============================================================

def train_model(opt_name, learning_rate=0.001, epochs=30):

    # Récupérer le nombre de features
    batch = next(iter(train_loader))
    input_size = batch["features"].shape[1]

    model = MLP(
        input_size=input_size,
        hidden_size=128
    ).to(device)

    criterion = nn.BCELoss()

    # Choix de l'optimiseur
    if opt_name == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )

    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    elif opt_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

    writer = SummaryWriter(
        f"runs/cardio_{opt_name}_lr{learning_rate}"
    )

    # Entraînement
    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, targets)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        # Loss moyenne
        epoch_loss = running_loss / len(train_loader)

        print(
            f"{opt_name} - "
            f"Epoch {epoch + 1}/{epochs} - "
            f"Loss: {epoch_loss:.4f}"
        )

        # TensorBoard
        writer.add_scalar(
            "Training Loss",
            epoch_loss,
            epoch
        )

    writer.close()

    # Retourner le modèle entraîné
    return model


# ============================================================
# Évaluation avec les métriques
# ============================================================

def evaluate_model(model, test_loader):

    model.eval()

    all_targets = []
    all_preds_probs = []

    with torch.no_grad():

        for batch in test_loader:

            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            outputs = model(inputs)

            # Stocker les probabilités et les cibles
            all_targets.extend(targets.cpu().numpy())
            all_preds_probs.extend(outputs.cpu().numpy())

    # Conversion en numpy array
    all_targets = np.array(all_targets)
    all_preds_probs = np.array(all_preds_probs)

    # Prédictions binaires avec un seuil de 0.5
    all_preds_classes = (all_preds_probs > 0.5).astype(int)

    # Calcul des métriques
    precision = precision_score(
        all_targets,
        all_preds_classes
    )

    recall = recall_score(
        all_targets,
        all_preds_classes
    )

    f1 = f1_score(
        all_targets,
        all_preds_classes
    )

    auc = roc_auc_score(
        all_targets,
        all_preds_probs
    )

    print(
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f} | "
        f"AUC: {auc:.4f}"
    )


# ============================================================
# Lancement des expériences
# ============================================================

models = {}

for opt in ["SGD", "Momentum", "RMSprop", "Adam"]:

    models[opt] = train_model(
        opt,
        learning_rate=0.001,
        epochs=30
    )


# ============================================================
# Évaluation du meilleur modèle
# ============================================================

# À remplacer par l'optimiseur que vous avez identifié
# comme meilleur lors de la comparaison TensorBoard.
best_model = models["Adam"]

evaluate_model(
    best_model,
    test_loader
)
