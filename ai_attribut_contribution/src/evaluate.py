# src/evaluate.py

import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_model(model, X_test, y_test, label_encoder, show_details=True):
    model.eval()
    # 1. Conversion des données en tenseurs
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

    # 2. Prédictions
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = outputs.argmax(dim=1).numpy()

    y_true = np.array(y_test)

    # 3. Score de précision
    acc = accuracy_score(y_true, predicted)
    print(f"\nPrécision sur le jeu de test : {acc*100:.2f}%")

    # 4. Matrice de confusion
    cm = confusion_matrix(y_true, predicted)
    classes = label_encoder.classes_

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Matrice de confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité")
    plt.tight_layout()
    # plt.show()
    # Save the figure
    plt.savefig("results/visualisations/confusion_matrix.png")
    print("Matrice de confusion sauvegardée dans results/visualisations/confusion_matrix.png")

    plt.close()

    # 5. Rapport de classification
    if show_details:
        print("\nRapport de classification :")
        print(classification_report(y_true, predicted, target_names=classes))

    return predicted  # utile pour analyse locale plus tard

