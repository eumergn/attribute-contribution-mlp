# src/train.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os
# from src.evaluate import plot_training_curve

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, output_dim)
        )

    def forward(self, x):
        return self.model(x)


def train_mlp(X_train, y_train, X_val, y_val, input_dim, output_dim, epochs=50, batch_size=32, lr=0.01):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt

    model = MLP(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.long)),
                              batch_size=batch_size, shuffle=True)

    val_inputs = torch.tensor(X_val, dtype=torch.float32)
    val_labels = torch.tensor(y_val, dtype=torch.long)

    train_accuracies = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = outputs.max(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_accuracy = correct / total
        train_accuracies.append(train_accuracy)

        model.eval()
        with torch.no_grad():
            outputs_val = model(val_inputs)
            preds_val = outputs_val.argmax(dim=1)
            acc_val = (preds_val == val_labels).float().mean().item()
        val_accuracies.append(acc_val)

        print(f"Époque {epoch+1}/{epochs} - Précision entraînement: {train_accuracy*100:.2f}% - Précision validation: {acc_val*100:.2f}%")

    # === Graphique des précisions uniquement ===
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), [a * 100 for a in train_accuracies], label="Exactitude Entraînement", marker='o', color='tab:blue')
    plt.plot(range(1, epochs + 1), [a * 100 for a in val_accuracies], label="Exactitude Validation", marker='s', color='tab:orange')
    plt.xlabel("Époque")
    plt.ylabel("Exactitude")
    plt.title("Courbes d'Exactitude d'Entraînement et de Validation")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("results/visualisations/train_val_curve.png")
    print("Graphique sauvegardé : results/visualisations/train_val_curve.png")

    torch.save(model.state_dict(), "results/models/model.pt")
    return model
