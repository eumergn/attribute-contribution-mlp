from src.pre_process import load_and_prepare_data
from src.train import train_mlp
from src.evaluate import evaluate_model
from src.explain_local import (
    select_instances_for_local_analysis,
    generate_all_perturbations,
    analyse_locale_complete
)

if __name__ == "__main__":
    # Chargement des données
    X_train, X_test, y_train, y_test, preprocessor, label_encoder = load_and_prepare_data("data/iris_extended.csv")

    # Entraînement du modèle
    model = train_mlp(X_train, y_train, X_test, y_test, input_dim=X_train.shape[1], output_dim=len(label_encoder.classes_))
    # # print("\n--- C.1 : Prédictions d'un batch de 4 instances ---")
    # # model.eval()
    # # X_batch = torch.tensor(X_test[:4], dtype=torch.float32)
    # # with torch.no_grad():
    # #     outputs = model(X_batch)
    # #     probs = torch.softmax(outputs, dim=1).numpy()
    # #     preds = outputs.argmax(dim=1).numpy()

    # # for i in range(4):
    # #     label = label_encoder.inverse_transform([preds[i]])[0]
    # #     proba_str = ', '.join([
    # #         f"{label_encoder.inverse_transform([j])[0]}: {100 * probs[i][j]:.1f}%"
    # #         for j in range(len(probs[i]))
    # #     ])
    # #     print(f"Instance {i} → Prédite : {label} | Probabilités : {proba_str}")
    # # Evaluate the model
    # evaluate_model(model, X_test, y_test, label_encoder)
    # Évaluation globale
    evaluate_model(model, X_test, y_test, label_encoder)

    # Sélection d'instances pour analyse locale
    selected_X, selected_y, selected_preds = select_instances_for_local_analysis(model, X_test, y_test, label_encoder)

    # Analyse avec amplitude = 0.1 (±10 %)
    all_perturbed_10 = generate_all_perturbations(selected_X, amplitude=0.1)
    analyse_locale_complete(model, all_perturbed_10["perturbations"], selected_preds, preprocessor, label_encoder, suffix="amp10")

    # Analyse avec amplitude = 0.2 (±20 %)
    all_perturbed_20 = generate_all_perturbations(selected_X, amplitude=0.2)
    analyse_locale_complete(model, all_perturbed_20["perturbations"], selected_preds, preprocessor, label_encoder, suffix="amp20")

