import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def preprocess_data(df: pd.DataFrame, target_col: str, test_size: float = 0.2, random_state: int = 42):
    """
    Nettoie et prépare les données pour l'entraînement de notre modèle.
    """
    # On drop la colonne ID car elle n'apporte aucune aide pour discriminr les classes
    df = df.drop('ID', axis=1)

    # On drop la colonne UNASSIGNED trop fortement correllée à la feature cible
    df = df.drop('UNASSIGNED', axis=1)

    # On drop la colonne WIDTH qui est redondante avec la feature NUMBER of request
    df = df.drop('WIDTH', axis=1)

    # On drop la colonne PENALTY qui semble être une pénalité interne au site
    df = df.drop('PENALTY', axis=1)

    # Les colonnes avec >25% de valeurs manquantes sont supprimées car l'imputation serait trop risquée et pourrait introduire du bruit.
    cols_to_drop = ['STANDARD_DEVIATION', 'SF_REFERRER', 'SF_FILETYPE']
    df = df.drop(cols_to_drop, axis=1)

    # Pour les autres colonnes, on effectue une imputation (médiane) par défaut
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            
    # Supprimer les doublons
    df.drop_duplicates(inplace=True)

    # Séparation des features et de la cible
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # Séparation en jeux d'entraînement et de test
    # On conserve la même proportion de classes avec stratify=y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':

    df = pd.read_csv('../data/data.csv')
    X_train, X_test, y_train, y_test = preprocess_data(df, 'ROBOT')
    print("Preprocessing terminé.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")