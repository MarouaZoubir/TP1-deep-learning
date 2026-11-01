
# ============================================================
# IMPORTS
# ============================================================

import os
import json
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ============================================================
# 1. REPRODUCTIBILITE ET CONFIGURATION
# ============================================================

torch.manual_seed(42)
random.seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/mnt/ssd/irishman"
SEQ_LENGTH = 500

print(f"Device utilise : {device}", flush=True)


# ============================================================
# 2. CHARGEMENT DES DONNEES
# ============================================================

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Nombre de chansons Train : {len(train_data)}", flush=True)
print(f"Nombre de chansons Val   : {len(val_data)}", flush=True)


# ============================================================
# 3. CONSTRUCTION DU VOCABULAIRE
# ============================================================

all_text = "".join(
    song["abc notation"] for song in train_data + val_data
)

unique_chars = sorted(set(all_text))
vocab_size = len(unique_chars)

print(f"Taille du vocabulaire : {vocab_size}", flush=True)


# ============================================================
# 4. MAPPINGS CARACTERE <-> INDICE
# ============================================================

char_to_index = {
    char: idx for idx, char in enumerate(unique_chars)
}

index_to_char = {
    idx: char for idx, char in enumerate(unique_chars)
}


def vectorize_string(s):
    return [char_to_index[char] for char in s]


# ============================================================
# 5. STATISTIQUES DE LONGUEUR
# ============================================================

lengths = [
    len(song["abc notation"]) for song in train_data
]

max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)

print(f"Longueur moyenne : {avg_len:.1f}", flush=True)
print(f"Longueur maximale : {max_len}", flush=True)


# ============================================================
# 6. PADDING ET TRONCATURE
# ============================================================

def pad_or_truncate(sequence, max_length=SEQ_LENGTH):

    if len(sequence) < max_length:
        return sequence + " " * (max_length - len(sequence))

    return sequence[:max_length]


# ============================================================
# 7. DATASET
# ============================================================

class MusicDataset(Dataset):

    def __init__(
        self,
        raw_data,
        seq_length=SEQ_LENGTH,
        subset_size=None
    ):

        if subset_size is not None:
            raw_data = raw_data[:subset_size]

        self.data = [
            vectorize_string(
                pad_or_truncate(
                    song["abc notation"],
                    seq_length
                )
            )
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        seq = self.data[idx]

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        return x, y


# ============================================================
# 8. MODELE RNN / LSTM
# ============================================================

class MusicModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_size=1024,
        rnn_type="lstm"
    ):

        super().__init__()

        self.rnn_type = rnn_type.lower()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        if self.rnn_type == "lstm":

            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                batch_first=True
            )

        elif self.rnn_type == "rnn":

            self.rnn = nn.RNN(
                embedding_dim,
                hidden_size,
                batch_first=True
            )

        else:
            raise ValueError(
                "rnn_type doit etre 'rnn' ou 'lstm'"
            )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x, hidden_state=None):

        embeds = self.embedding(x)

        out, hidden_state = self.rnn(
            embeds,
            hidden_state
        )

        logits = self.fc(out)

        return logits, hidden_state


# ============================================================
# 9. COMPTAGE DES PARAMETRES
# ============================================================

def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# 10. FONCTION D'EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(model, val_loader, loss_fn, vocab_size):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in val_loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits, _ = model(x)

        loss = loss_fn(
            logits.reshape(-1, vocab_size),
            y.reshape(-1)
        )

        total_loss += loss.item() * y.numel()

        preds = logits.argmax(dim=-1)

        total_correct += (preds == y).sum().item()
        total_tokens += y.numel()

    return (
        total_loss / total_tokens,
        total_correct / total_tokens
    )


# ============================================================
# 11. FONCTION D'ENTRAINEMENT
# ============================================================

def train_model(
    rnn_type="lstm",
    subset_size=50000,
    num_epochs=30,
    batch_size=128,
    lr=5e-3,
    embedding_dim=256,
    hidden_size=1024,
    max_grad_norm=1.0,
    patience=30
):

    # --------------------------------------------------------
    # Preparation des datasets
    # --------------------------------------------------------

    train_dataset = MusicDataset(
        train_data,
        subset_size=subset_size
    )

    val_dataset = MusicDataset(val_data)

    print(
        f"\nNombre de chansons utilisees pour Train : "
        f"{len(train_dataset)}",
        flush=True
    )

    print(
        f"Nombre de chansons utilisees pour Val : "
        f"{len(val_dataset)}",
        flush=True
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # --------------------------------------------------------
    # Initialisation du modele
    # --------------------------------------------------------

    model = MusicModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        rnn_type=rnn_type
    ).to(device)

    print(
        f"\nNombre de parametres du modele : "
        f"{count_parameters(model):,}",
        flush=True
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    loss_fn = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Configuration TensorBoard
    # --------------------------------------------------------

    run_name = (
        f"{rnn_type}_sub{subset_size}_"
        f"hidden{hidden_size}_epochs{num_epochs}_"
        f"clip{max_grad_norm}_{time.strftime('%H%M%S')}"
    )

    os.makedirs("runs_rnn", exist_ok=True)

    writer = SummaryWriter(
        log_dir=f"runs_rnn/{run_name}"
    )

    save_path = f"best_{run_name}.pth"

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    # --------------------------------------------------------
    # Boucle d'entrainement
    # --------------------------------------------------------

    for epoch in range(1, num_epochs + 1):

        model.train()

        total_train_loss = 0.0
        total_train_tokens = 0

        epoch_grad_norms = []

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{num_epochs}",
            leave=True
        )

        for x, y in progress_bar:

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward
            logits, _ = model(x)

            loss = loss_fn(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )

            # Backward
            loss.backward()

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_grad_norm
            )

            epoch_grad_norms.append(grad_norm.item())

            # TensorBoard : norme des gradients
            if global_step % 20 == 0:

                writer.add_scalar(
                    "GradNorm/step",
                    grad_norm.item(),
                    global_step
                )

            # Mise a jour des poids
            optimizer.step()

            total_train_loss += loss.item() * y.numel()
            total_train_tokens += y.numel()

            global_step += 1

        # ----------------------------------------------------
        # Evaluation a la fin de chaque epoque
        # ----------------------------------------------------

        avg_train_loss = (
            total_train_loss / total_train_tokens
        )

        avg_grad_norm = (
            sum(epoch_grad_norms) / len(epoch_grad_norms)
        )

        val_loss, val_acc = evaluate_model(
            model,
            val_loader,
            loss_fn,
            vocab_size
        )

        # ----------------------------------------------------
        # Enregistrement TensorBoard
        # ----------------------------------------------------

        writer.add_scalar(
            "Loss/Train",
            avg_train_loss,
            epoch
        )

        writer.add_scalar(
            "Loss/Validation",
            val_loss,
            epoch
        )

        writer.add_scalar(
            "Accuracy/Validation",
            val_acc,
            epoch
        )

        writer.add_scalar(
            "GradNorm/Epoch_Avg",
            avg_grad_norm,
            epoch
        )

        writer.flush()

        print(
            f"\nEpoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"Avg GradNorm: {avg_grad_norm:.3f}",
            flush=True
        )

        # ----------------------------------------------------
        # Early stopping et sauvegarde du meilleur modele
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"Meilleur modele sauvegarde : {save_path}",
                flush=True
            )

        else:

            patience_counter += 1

            if patience_counter >= patience:

                print(
                    f"Early stopping a l'epoque {epoch}",
                    flush=True
                )

                break

    writer.close()

    # --------------------------------------------------------
    # Rechargement du meilleur modele
    # --------------------------------------------------------

    model.load_state_dict(
        torch.load(
            save_path,
            map_location=device
        )
    )

    print(
        f"\nMeilleur modele recharge : {save_path}",
        flush=True
    )

    return model


# ============================================================
# 12. GENERATION MUSICALE AVEC CACHE DE L'ETAT CACHE
# ============================================================

@torch.no_grad()
def generate_music(
    model,
    start_sequence,
    length=300,
    temperature=1.0
):

    if temperature <= 0:
        raise ValueError(
            "La temperature doit etre strictement positive."
        )

    model.eval()

    # Vectorisation du prompt
    input_ids = []

    for char in start_sequence:

        if char not in char_to_index:
            raise ValueError(
                f"Le caractere {repr(char)} est absent du vocabulaire."
            )

        input_ids.append(char_to_index[char])

    input_tensor = torch.tensor(
        input_ids,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    generated = start_sequence

    # Traitement initial du prompt
    logits, hidden_state = model(input_tensor)

    # Generation caractere par caractere
    for _ in range(length):

        next_logits = logits[:, -1, :]

        # Application de la temperature
        probabilities = F.softmax(
            next_logits / temperature,
            dim=-1
        )

        # Echantillonnage du prochain caractere
        next_char_id = torch.multinomial(
            probabilities,
            num_samples=1
        )

        next_char = index_to_char[
            next_char_id.item()
        ]

        generated += next_char

        # Cache de l'etat cache :
        # on ne traite que le nouveau caractere
        logits, hidden_state = model(
            next_char_id,
            hidden_state
        )

    return generated


# ============================================================
# 13. ENTRAINEMENT GRANDE ECHELLE : LSTM
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60, flush=True)
    print("ENTRAINEMENT LSTM A GRANDE ECHELLE", flush=True)
    print("=" * 60, flush=True)

    model_lstm = train_model(
        rnn_type="lstm",
        subset_size=50000,
        num_epochs=30,
        batch_size=128,
        lr=5e-3,
        embedding_dim=256,
        hidden_size=1024,
        max_grad_norm=1.0,
        patience=30
    )

    print("\nEntrainement termine.", flush=True)

    # ========================================================
    # 14. GENERATION DES TROIS PARTITIONS
    # ========================================================

    prompt = (
        "X:1\n"
        "T:Irish Reel\n"
        "M:4/4\n"
        "L:1/8\n"
        "K:D\n"
    )

    temperatures = [0.2, 0.8, 2.0]

    print("\n" + "=" * 60, flush=True)
    print("GENERATION MUSICALE", flush=True)
    print("=" * 60, flush=True)

    for temperature in temperatures:

        print(
            f"\n\n========== TEMPERATURE {temperature} ==========",
            flush=True
        )

        generated_music = generate_music(
            model=model_lstm,
            start_sequence=prompt,
            length=300,
            temperature=temperature
        )

        # Affichage dans le fichier de log Slurm
        print(
            generated_music,
            flush=True
        )

        # Sauvegarde dans un fichier ABC
        filename = (
            f"music_T{str(temperature).replace('.', '')}.abc"
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(generated_music)

        print(
            f"Partition sauvegardee : {filename}",
            flush=True
        )

    print(
        "\n========== JOB TERMINE AVEC SUCCES ==========",
        flush=True
    )

