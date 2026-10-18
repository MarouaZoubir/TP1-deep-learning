import os
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ============================================================
# Configuration
# ============================================================

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ============================================================
# Exercice 1 : transformations ImageNet
# ============================================================

# Entraînement : augmentation de données
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])

# Validation : transformations déterministes
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    ),
])


def get_dataloaders(batch_size=32, n_workers=1, subset_size=None):
    """Charge ImageNet et retourne les DataLoaders."""

    print("Chargement d'ImageNet...")

    train_dataset = ImageNet(
        root=data_dir,
        split="train",
        transform=train_transform
    )

    val_dataset = ImageNet(
        root=data_dir,
        split="val",
        transform=val_transform
    )

    # Sous-ensembles déterministes pour les expériences
    if subset_size is not None:
        generator = torch.Generator().manual_seed(42)

        train_size = min(subset_size, len(train_dataset))
        val_size = min(max(1, subset_size // 4), len(val_dataset))

        train_indices = torch.randperm(
            len(train_dataset),
            generator=generator
        )[:train_size]

        val_indices = torch.randperm(
            len(val_dataset),
            generator=generator
        )[:val_size]

        train_dataset = Subset(
            train_dataset,
            train_indices.tolist()
        )

        val_dataset = Subset(
            val_dataset,
            val_indices.tolist()
        )

    print(f"Images d'entraînement : {len(train_dataset)}")
    print(f"Images de validation : {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda")
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda")
    )

    return train_loader, val_loader


# ============================================================
# Exercice 2 : MLP
# ============================================================

class MLP(nn.Module):
    def __init__(
        self,
        input_size=3 * 227 * 227,
        hidden_size=256,
        num_classes=1000
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# Exercice 2 : AlexNet
# ============================================================

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                3, 64,
                kernel_size=11,
                stride=4,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                64, 192,
                kernel_size=5,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(
                192, 384,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                384, 256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256, 256,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AlexNetDeep(nn.Module):
    """AlexNet avec trois couches convolutives supplémentaires après Conv5."""

    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Trois nouvelles convolutions 256 -> 256, kernel 3x3, padding 1
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Pooling d'AlexNet conservé après les convolutions
            nn.MaxPool2d(kernel_size=3, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# Comptage des paramètres
# ============================================================

def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


# ============================================================
# Évaluation
# ============================================================

@torch.no_grad()
def evaluate(loader, model, criterion):
    """Évalue le modèle et retourne la loss et l'accuracy."""

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        loader,
        desc="Validation",
        leave=False
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)

        running_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += batch_size

    if total == 0:
        return 0.0, 0.0

    average_loss = running_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ============================================================
# Exercice 3 : entraînement avec Early Stopping
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=100,
    lr=1e-3,
    patience=5,
    output_dir="checkpoints",
    model_name="alexnet",
    subset_size=None
):
    os.makedirs(output_dir, exist_ok=True)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )

    # Un dossier TensorBoard distinct pour chaque expérience
    run_tag = f"{model_name}_sub{subset_size}_lr{lr}"
    writer = SummaryWriter(
        log_dir=os.path.join("runs", run_tag)
    )

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = os.path.join(
        output_dir,
        f"{run_tag}_best.pth"
    )

    try:
        for epoch in range(epochs):
            model.train()

            running_loss = 0.0
            total_correct = 0
            total_samples = 0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{epochs}"
            )

            for images, labels in progress_bar:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                batch_size = labels.size(0)

                running_loss += loss.item() * batch_size

                predictions = outputs.argmax(dim=1)
                total_correct += (
                    predictions == labels
                ).sum().item()

                total_samples += batch_size

                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}"
                )

            # Métriques d'entraînement
            train_loss = running_loss / max(total_samples, 1)

            train_accuracy = (
                100.0 * total_correct / max(total_samples, 1)
            )

            # Attention : l'ordre des arguments est corrigé
            val_loss, val_accuracy = evaluate(
                val_loader,
                model,
                criterion
            )

            val_accuracy_percent = 100.0 * val_accuracy

            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train loss: {train_loss:.4f} | "
                f"Train acc: {train_accuracy:.2f}% | "
                f"Val loss: {val_loss:.4f} | "
                f"Val acc: {val_accuracy_percent:.2f}%"
            )

            # TensorBoard
            writer.add_scalar(
                "Loss/train",
                train_loss,
                epoch + 1
            )

            writer.add_scalar(
                "Loss/validation",
                val_loss,
                epoch + 1
            )

            writer.add_scalar(
                "Accuracy/train",
                train_accuracy,
                epoch + 1
            )

            writer.add_scalar(
                "Accuracy/validation",
                val_accuracy_percent,
                epoch + 1
            )

            writer.flush()

            # Early Stopping : sauvegarder le meilleur modèle
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0

                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                        "model_name": model_name,
                        "subset_size": subset_size,
                        "learning_rate": lr
                    },
                    best_model_path
                )

                print(
                    f"Meilleur modèle sauvegardé : "
                    f"{best_model_path}"
                )

            else:
                epochs_without_improvement += 1

                print(
                    f"Pas d'amélioration : "
                    f"{epochs_without_improvement}/{patience}"
                )

                if epochs_without_improvement >= patience:
                    print(
                        f"Early Stopping déclenché "
                        f"à l'époque {epoch + 1}."
                    )
                    break

    finally:
        writer.close()

    print(f"Meilleure époque : {best_epoch}")
    print(f"Meilleure val loss : {best_val_loss:.4f}")

    return model


# ============================================================
# Point d'entrée
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="TP3 : classification ImageNet"
    )

    parser.add_argument(
        "--model",
        choices=["mlp", "alexnet", "alexnet_deep"],
        default="alexnet",
        help="Architecture à entraîner : mlp, alexnet ou alexnet_deep"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Nombre maximal d'époques"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille du batch"
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Nombre de workers du DataLoader"
    )

    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Nombre d'images d'entraînement pour un test rapide"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate"
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Patience de l'Early Stopping"
    )

    parser.add_argument(
        "--output-dir",
        default="checkpoints",
        help="Répertoire des checkpoints"
    )

    args = parser.parse_args()

    print(f"Device : {device}")

    if device.type == "cuda":
        print(f"GPU : {torch.cuda.get_device_name(0)}")

    # Création du modèle choisi
    if args.model == "mlp":
        model = MLP()
    elif args.model == "alexnet":
        model = AlexNet()
    else:
        model = AlexNetDeep()

    print(f"Modèle : {args.model}")
    print(
        f"Nombre de paramètres : "
        f"{count_parameters(model):,}"
    )

    # Vérification de la forme de sortie
    model.eval()

    with torch.no_grad():
        test_input = torch.randn(1, 3, 227, 227)
        test_output = model(test_input)

    print(f"Forme de sortie : {test_output.shape}")

    # Chargement des données
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        n_workers=args.workers,
        subset_size=args.subset_size
    )

    # Entraînement
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        output_dir=args.output_dir,
        model_name=args.model,
        subset_size=args.subset_size
    )


if __name__ == "__main__":
    main()
