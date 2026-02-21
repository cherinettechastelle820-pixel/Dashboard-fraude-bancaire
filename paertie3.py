# ======================================================
# PARTIE 3 – EXPÉRIMENTATION SUR UN NOUVEAU JEU DE DONNÉES
# Détection de fraude bancaire
# ======================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

# =========================
# 3.1 Chargement des données
# =========================

df = pd.read_csv("fraude_bancaire.csv")

print("Aperçu des 5 premières lignes :")
print(df.head())

print("\nDimensions :", df.shape)
print("\nColonnes :", df.columns)

# =========================
# 3.2 Variable cible
# =========================

print("\nRépartition des classes :")
print(df["fraude"].value_counts())

# =========================
# 3.3 Séparation X / y
# =========================

y = df["fraude"]
X = df.drop("fraude", axis=1)

# =========================
# 3.4 Encodage des variables catégorielles
# =========================

cat_cols = X.select_dtypes(include="object").columns
X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("\nDimensions après encodage :", X.shape)

# =========================
# 3.5 Imputation des valeurs manquantes
# =========================

imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

print("Valeurs manquantes restantes :", X.isna().sum().sum())

# =========================
# 3.6 Split apprentissage / test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================
# 3.7 Normalisation
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 3.8 Apprentissage du modèle
# =========================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

# =========================
# 3.9 Évaluation
# =========================

print("\nScore apprentissage :", rf.score(X_train, y_train))
print("Score test :", rf.score(X_test, y_test))

y_pred = rf.predict(X_test)

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

print("\n===== FIN PARTIE 3 =====")