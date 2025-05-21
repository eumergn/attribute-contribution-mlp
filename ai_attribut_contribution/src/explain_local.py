# src/explain_local.py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def select_instances_for_local_analysis(model, X_test, y_test, label_encoder, num_correct=3, num_incorrect=3):
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # 1. Prédictions
    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = outputs.argmax(dim=1)

    # 2. Corrects / incorrects
    correct_mask = (predicted == y_test_tensor)
    incorrect_mask = ~correct_mask

    correct_indices = torch.where(correct_mask)[0][:num_correct]
    incorrect_indices = torch.where(incorrect_mask)[0][:num_incorrect]

    selected_indices = torch.cat([correct_indices, incorrect_indices])

    # 3. Données sélectionnées
    selected_X = X_test_tensor[selected_indices]
    selected_y = y_test_tensor[selected_indices]
    selected_preds = predicted[selected_indices]

    # 4. Affichage pour vérification
    print("\nInstances sélectionnées pour l'analyse locale :")
    for i, idx in enumerate(selected_indices):
        vrai = label_encoder.inverse_transform([selected_y[i].item()])[0]
        predit = label_encoder.inverse_transform([selected_preds[i].item()])[0]
        status = "Correct" if selected_y[i] == selected_preds[i] else "Faux"
        print(f"[{i}]  Classe réelle : {vrai:<10} | Prédite : {predit:<10} | {status}")

    return selected_X.numpy(), selected_y.numpy(), selected_preds.numpy()


def explain_with_local_linear_model(model, perturbed_data, original_class, label_encoder):
    """
    Entraîne un modèle linéaire localement pour approximer le comportement du réseau
    perturbed_data : np.array de perturbations (n, d)
    original_class : int (classe réelle de l'instance)
    Retourne : importance des features (coefs du modèle)
    """
    # Prédictions du réseau sur les perturbations
    model.eval()
    inputs = torch.tensor(perturbed_data, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(inputs)
        preds = outputs.argmax(dim=1).numpy()

    # Entraîner un modèle linéaire sur les perturbations
    clf = LogisticRegression()
    clf.fit(perturbed_data, preds)

    # Importance des features pour la classe prédite
    class_index = original_class
    coefs = clf.coef_[class_index] if clf.coef_.shape[0] > 1 else clf.coef_[0]

    print("\nImportance des attributs (approx. locale) :")
    for i, coef in enumerate(coefs):
        print(f"Attribut {i} : {coef:.4f}")

    return coefs


def train_local_linear_model(X_local, y_local, lr=0.1, epochs=1000):
    """
    Entraîne une régression logistique binaire (softmax linéaire) avec descente de gradient
    X_local : (n_samples, n_features)
    y_local : (n_samples,) -> classes entières
    """
    from sklearn.preprocessing import OneHotEncoder

    if len(np.unique(y_local)) < 2:
        print("⚠️ Impossible d'entraîner un modèle local : une seule classe présente dans les prédictions.")
        return None, None

    n_samples, n_features = X_local.shape
    classes = np.unique(y_local)
    n_classes = len(classes)

    # Initialiser les poids
    W = np.zeros((n_features, n_classes))
    b = np.zeros(n_classes)

    # One-hot encode les y
    encoder = OneHotEncoder(sparse_output=False, categories='auto')
    y_onehot = encoder.fit_transform(y_local.reshape(-1, 1))

    for _ in range(epochs):
        # Forward : prédiction (softmax)
        logits = np.dot(X_local, W) + b
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Gradient
        error = probs - y_onehot
        grad_W = np.dot(X_local.T, error) / n_samples
        grad_b = np.mean(error, axis=0)

        # Update
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b


def train_local_tree_model(X_local, y_local, max_depth=3):
    tree = DecisionTreeClassifier(max_depth=max_depth)
    tree.fit(X_local, y_local)
    return tree



def plot_contributions(W, classe_index, feature_names, instance_index, save=False, suffix=""):
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    if W.shape[1] == 1:
        contributions = W[:, 0]
    else:
        contributions = W[:, classe_index]

    indices_sorted = np.argsort(np.abs(contributions))[::-1]
    contributions = contributions[indices_sorted]
    feature_names = [feature_names[i] for i in indices_sorted]

    N = min(10, len(contributions))
    contributions = contributions[:N]
    feature_names = feature_names[:N]

    plt.figure(figsize=(10, 4))
    bars = plt.bar(range(len(contributions)), contributions)
    plt.xticks(ticks=range(len(contributions)), labels=feature_names, rotation=45, ha='right')
    plt.ylabel("Importance (poids)")
    plt.title(f"Contributions locales - Instance {instance_index} - Classe prédite {classe_index}")
    plt.tight_layout()

    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(i, yval, f"{yval:.2f}", ha='center', va='bottom', fontsize=8)

    if save:
        os.makedirs("results/visualisations/", exist_ok=True)
        filename = f"instance_{instance_index}_{suffix}.png" if suffix else f"instance_{instance_index}.png"
        path = os.path.join("results/visualisations", filename)
        plt.savefig(path)
        print(f"Sauvegardé : {path}")
        plt.close()
    else:
        plt.show()




def analyse_locale_complete(model, all_perturbed, selected_preds, preprocessor, label_encoder, feature_names=None, suffix=""):
    if feature_names is None:
        feature_names = preprocessor.get_feature_names_out()

    for i, perturbed in enumerate(all_perturbed):
        print(f"\nAnalyse de l'instance {i}")

        y_local = predict_perturbations(model, perturbed)
        unique, counts = np.unique(y_local, return_counts=True)

        print(f"  ➤ Prédictions sur perturbations :")
        for label, count in zip(unique, counts):
            class_name = label_encoder.inverse_transform([label])[0]
            print(f"    {class_name} : {count} fois")

        if len(unique) < 2:
            print("     Instance ignorée : aucune diversité dans les prédictions.")
            continue

        W, b = train_local_linear_model(perturbed, y_local)
        classe_cible = selected_preds[i]
        unique_classes = np.unique(y_local)

        if classe_cible not in unique_classes:
            print(f"    ⚠️ Classe prédite {classe_cible} absente dans y_local → ignorée.")
            continue

        local_class_index = np.where(unique_classes == classe_cible)[0][0]

        plot_contributions(W, local_class_index, feature_names, instance_index=i, save=True, suffix=suffix)



def predict_perturbations(model, X_perturbed):
    """
    Utilitaire pour prédire les classes des données perturbées.
    """
    model.eval()
    X_tensor = torch.tensor(X_perturbed, dtype=torch.float32)
    with torch.no_grad():
        outputs = model(X_tensor)
        preds = outputs.argmax(dim=1).numpy()
    return preds


# Generation des perturbations
def generate_perturbations(instance, num_perturbations=250, amplitude=0.1, seed=None):
    """
    Génère des perturbations aléatoires autour d'une instance normalisée
    instance : vecteur numpy (1D)
    Retourne : matrice (250, nb_features)
    """
    if seed is not None:
        np.random.seed(seed)

    d = instance.shape[0]
    perturbations = []

    for _ in range(num_perturbations):
        bruit = np.random.uniform(-amplitude, amplitude, size=d)
        instance_bruitee = instance * (1 + bruit)
        perturbations.append(instance_bruitee)

    return np.array(perturbations)

# Generation de toutes les perturbations des 6 instances
def generate_all_perturbations(selected_X, n_perturbations=250, amplitude=0.2):
    all_perturbed = []
    for i, instance in enumerate(selected_X):
        perturbed = generate_perturbations(instance, num_perturbations=n_perturbations)
        all_perturbed.append(perturbed)
        print(f"Instance {i} : {perturbed.shape[0]} perturbations générées.")
    return {
    "perturbations": all_perturbed,
    "originals": selected_X
    }

