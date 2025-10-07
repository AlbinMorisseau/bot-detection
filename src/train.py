import pandas as pd
import xgboost as xgb
import optuna
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from preprocess import preprocess_data

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Fonction objectif qu'Optuna essaiera de maximiser.

    """
    # hyper paramètres à optimiser
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1, 10),
        'alpha': trial.suggest_float('alpha', 0, 5)
    }

    # Gestion du déséquilibre de classes
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params['scale_pos_weight'] = scale_pos_weight

    model = xgb.XGBClassifier(**params, early_stopping_rounds=50,random_state=42, n_jobs=-1)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Prédictions probabilistes pour PR AUC
    y_proba = model.predict_proba(X_val)[:,1]
    pr_auc = average_precision_score(y_val, y_proba)

    return pr_auc

def train_and_evaluate():
    # Chargement et prétraitement
    print("Chargement et prétraitement des données...")
    df = pd.read_csv('../data/data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df, 'ROBOT')

    # Optimisation des hyperparamètres
    print("Début de l'optimisation des hyperparamètres avec Optuna...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda trial: objective(trial, X_train, y_train, X_test, y_test), n_trials=50)

    print(f"\nMeilleur PR AUC (validation): {study.best_value:.4f}")
    print(f"Meilleurs hyperparamètres:\n{study.best_params}\n")

    # Entraînement du modèle final
    final_params = study.best_params
    final_params['objective'] = 'binary:logistic'
    final_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

    final_model = xgb.XGBClassifier(**final_params, early_stopping_rounds=50, random_state=42, n_jobs=-1)
    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False
    )

    # Évaluation sur le jeu de test
    print("\nÉvaluation du modèle final sur le jeu de test...\n")
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)[:,1]

    print(classification_report(y_test, y_pred, target_names=['Humain', 'Robot']))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Humain', 'Robot'], yticklabels=['Humain', 'Robot'])
    plt.xlabel('Prédiction')
    plt.ylabel('Vérité terrain')
    plt.title('Matrice de Confusion')
    plt.savefig('results/confusion_matrix.jpg')
    plt.show()

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.5f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig('results/ROC_curve.jpg')
    plt.show()

    # Courbe Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig('results/PR_curve.jpg')
    plt.show()

    # Analyse de l'importance des features
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 des features les plus importantes :\n", feature_importances.head(10))
    plt.figure(figsize=(12,8))
    sns.barplot(x='importance', y='feature', data=feature_importances.head(10))
    plt.title('Top 10 des features les plus importantes')
    plt.savefig('results/10_best_features.jpg')
    plt.show()

if __name__ == '__main__':
    train_and_evaluate()
