import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =========================
# 1. Chargement des données
# =========================

df = pd.read_csv("acs2017_census_tract_data 2.csv")

print("Aperçu des 5 premières lignes du jeu de données :")
print(df.head())

print("\nDimensions du jeu de données (lignes, colonnes) :")
print(df.shape)

print("\nInformations générales sur le jeu de données :")
print(df.info())

print("\nStatistiques descriptives des variables numériques :")
print(df.describe())

print("\nRépartition des valeurs de Income :")
print(df["Income"].describe())

# =========================
# 2. Nettoyage de la cible
# =========================

print("\nNombre de valeurs manquantes dans Income :")
print(df["Income"].isna().sum())

print("\nSuppression des lignes avec Income manquant...")
df = df.dropna(subset=["Income"])

print("Nettoyage terminé.")
print("Nouveau nombre total d'instances :", df.shape[0])

# =========================
# 3. Création de la variable cible (classification)
# =========================

median_income = df["Income"].median()
print("\nRevenu médian :", median_income)

df["Income_class"] = (df["Income"] > median_income).astype(int)

print("\nRépartition des classes (Income_class) :")
print(df["Income_class"].value_counts())

# =========================
# 4. Résumé des données
# =========================

print("\n===== RÉSUMÉ DES DONNÉES =====")
print(f"Nombre d'instances : {df.shape[0]}")
print(f"Nombre de variables : {df.shape[1]}")
print("Nombre de classes :", df["Income_class"].nunique())

print("\nTypes des variables :")
print(df.dtypes.value_counts())

# =========================
# 5. Création du dossier figures
# =========================

if not os.path.exists("figures"):
    os.makedirs("figures")
    print("\nDossier 'figures' créé.")
else:
    print("\nDossier 'figures' déjà existant.")

# =========================
# 6. Visualisations
# =========================

variables = ["TotalPop", "Income", "Poverty", "Unemployment"]

print("\nGénération du pairplot...")
pairplot = sns.pairplot(df[variables], diag_kind="kde")
pairplot.fig.suptitle("Pairplot des variables socio-économiques", y=1.02)
pairplot.savefig("figures/pairplot_variables.png")
plt.close()
print("Pairplot sauvegardé dans figures/pairplot_variables.png")

print("\nGénération du graphique Poverty vs Income")
plt.figure()
sns.regplot(x="Poverty", y="Income", data=df)
plt.title("Relation entre Poverty et Income")
plt.savefig("figures/poverty_vs_income.png")
plt.close()
print("Graphique sauvegardé : figures/poverty_vs_income.png")

print("\nGénération du graphique Unemployment vs Income")
plt.figure()
sns.regplot(x="Unemployment", y="Income", data=df)
plt.title("Relation entre Unemployment et Income")
plt.savefig("figures/unemployment_vs_income.png")
plt.close()
print("Graphique sauvegardé : figures/unemployment_vs_income.png")

# =========================
# 7. Séparation X / y
# =========================

y = df["Income_class"]
X = df.drop(["Income", "Income_class"], axis=1)

print("\nDimensions après nettoyage :")
print("X :", X.shape)
print("y :", y.shape)

# =========================
# 8. Split apprentissage / test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nSéparation réussie.")
print("Taille apprentissage :", X_train.shape[0])
print("Taille test :", X_test.shape[0])

# =========================
# 9. Encodage des variables catégorielles
# =========================

print("\nEncodage des variables catégorielles...")

cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

print("Variables catégorielles :")
print(list(cat_cols))

print("\nVariables numériques :")
print(list(num_cols))

X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

print("\nEncodage terminé.")
print("Nouvelles dimensions de X :", X_encoded.shape)

# =========================
# 10. Imputation des valeurs manquantes
# =========================

from sklearn.impute import SimpleImputer

print("\nImputation des valeurs manquantes (médiane)...")

imputer = SimpleImputer(strategy="median")

X_imputed = pd.DataFrame(
    imputer.fit_transform(X_encoded),
    columns=X_encoded.columns
)

print("Imputation terminée.")
print("Valeurs manquantes restantes :", X_imputed.isna().sum().sum())

# =========================
# 11. Nouveau split après imputation
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_imputed,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nNouveau split après imputation effectué.")
print("Train :", X_train.shape)
print("Test :", X_test.shape)
# =========================
# 11. Normalisation (OBLIGATOIRE pour KNN)
# =========================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nNormalisation des données effectuée.")
print("X_train_scaled :", X_train_scaled.shape)
print("X_test_scaled :", X_test_scaled.shape)

# =========================
# 12. Apprentissage KNN (k = 5)
# =========================

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

print("\nModèle KNN entraîné (k = 5).")

train_score = knn.score(X_train_scaled, y_train)
test_score = knn.score(X_test_scaled, y_test)

print("\n===== SCORES KNN =====")
print("Score apprentissage :", train_score)
print("Score test :", test_score)

# =========================
# 13. Matrice de confusion
# =========================

from sklearn.metrics import confusion_matrix, classification_report

y_pred = knn.predict(X_test_scaled)

print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred))

print("\nRapport de classification :")
print(classification_report(y_test, y_pred))

# =========================
# 14. Étude de l'influence du paramètre k
# =========================

print("\n===== ÉTUDE DE L'INFLUENCE DE k =====")

k_values = range(1, 21)
train_scores = []
test_scores = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train_scaled, y_train)
    
    train_scores.append(model.score(X_train_scaled, y_train))
    test_scores.append(model.score(X_test_scaled, y_test))
    
    print(f"k={k} | train={train_scores[-1]:.4f} | test={test_scores[-1]:.4f}")

# =========================
# 15. Sauvegarde de la figure k
# =========================

plt.figure()
plt.plot(k_values, train_scores, label="Score apprentissage")
plt.plot(k_values, test_scores, label="Score test")
plt.xlabel("Nombre de voisins (k)")
plt.ylabel("Score")
plt.title("Influence du paramètre k sur les performances du KNN")
plt.legend()

plt.savefig("figures/knn_influence_k.png")
plt.close()

print("\nFigure sauvegardée : figures/knn_influence_k.png")

# =========================
# 16. Arbre de décision (CART)
# =========================

from sklearn.tree import DecisionTreeClassifier

print("\n===== ARBRE DE DÉCISION (CART) =====")

cart = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

cart.fit(X_train, y_train)

print("Modèle CART entraîné.")
print("Profondeur maximale :", cart.get_depth())
print("Nombre de feuilles :", cart.get_n_leaves())

train_score_cart = cart.score(X_train, y_train)
test_score_cart = cart.score(X_test, y_test)

print("\nScores CART :")
print("Score apprentissage :", train_score_cart)
print("Score test :", test_score_cart)

# =========================
# 17. Évaluation CART
# =========================

from sklearn.metrics import confusion_matrix, classification_report

y_pred_cart = cart.predict(X_test)

print("\nMatrice de confusion (CART) :")
print(confusion_matrix(y_test, y_pred_cart))

print("\nRapport de classification (CART) :")
print(classification_report(y_test, y_pred_cart))

# =========================
# 18. Influence de la profondeur (max_depth)
# =========================

print("\n===== ÉTUDE DE max_depth =====")

depths = range(1, 21)
train_scores = []
test_scores = []

for d in depths:
    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=d,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))
    
    print(f"max_depth={d} | train={train_scores[-1]:.4f} | test={test_scores[-1]:.4f}")

# =========================
# 19. Sauvegarde figure CART
# =========================

plt.figure()
plt.plot(depths, train_scores, label="Score apprentissage")
plt.plot(depths, test_scores, label="Score test")
plt.xlabel("Profondeur maximale (max_depth)")
plt.ylabel("Score")
plt.title("Influence de la profondeur sur CART")
plt.legend()

plt.savefig("figures/cart_influence_depth.png")
plt.close()

print("\nFigure sauvegardée : figures/cart_influence_depth.png")

# =========================
# 20. Random Forest
# =========================

from sklearn.ensemble import RandomForestClassifier

print("\n===== RANDOM FOREST =====")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

print("Modèle Random Forest entraîné.")
print("Nombre d'arbres :", rf.n_estimators)

train_score_rf = rf.score(X_train, y_train)
test_score_rf = rf.score(X_test, y_test)

print("\nScores Random Forest :")
print("Score apprentissage :", train_score_rf)
print("Score test :", test_score_rf)

# =========================
# 21. Évaluation Random Forest
# =========================

from sklearn.metrics import confusion_matrix, classification_report

y_pred_rf = rf.predict(X_test)

print("\nMatrice de confusion (Random Forest) :")
print(confusion_matrix(y_test, y_pred_rf))

print("\nRapport de classification (Random Forest) :")
print(classification_report(y_test, y_pred_rf))

# =========================
# 22. Importance des variables
# =========================

importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Variable": X_train.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 des variables les plus importantes :")
print(importance_df.head(10))

# =========================
# 23. Sauvegarde figure importance Random Forest
# =========================

plt.figure(figsize=(8, 6))
plt.barh(
    importance_df.head(10)["Variable"],
    importance_df.head(10)["Importance"]
)
plt.xlabel("Importance")
plt.title("Top 10 des variables les plus importantes (Random Forest)")
plt.gca().invert_yaxis()

plt.savefig("figures/rf_feature_importance.png")
plt.close()

print("\nFigure sauvegardée : figures/rf_feature_importance.png")

# =========================
# 24. Gradient Boosting
# =========================

from sklearn.ensemble import GradientBoostingClassifier

print("\n===== GRADIENT BOOSTING =====")

gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

gb.fit(X_train, y_train)

print("Modèle Gradient Boosting entraîné.")
print("Nombre d'arbres :", gb.n_estimators)
print("Learning rate :", gb.learning_rate)

train_score_gb = gb.score(X_train, y_train)
test_score_gb = gb.score(X_test, y_test)

print("\nScores Gradient Boosting :")
print("Score apprentissage :", train_score_gb)
print("Score test :", test_score_gb)

# =========================
# 25. Évaluation Gradient Boosting
# =========================

from sklearn.metrics import confusion_matrix, classification_report

y_pred_gb = gb.predict(X_test)

print("\nMatrice de confusion (Gradient Boosting) :")
print(confusion_matrix(y_test, y_pred_gb))

print("\nRapport de classification (Gradient Boosting) :")
print(classification_report(y_test, y_pred_gb))

# =========================
# 26. Importance des variables (Gradient Boosting)
# =========================

gb_importances = gb.feature_importances_

gb_importance_df = pd.DataFrame({
    "Variable": X_train.columns,
    "Importance": gb_importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 des variables les plus importantes (Gradient Boosting) :")
print(gb_importance_df.head(10))

# =========================
# 27. Sauvegarde figure importance Gradient Boosting
# =========================

plt.figure(figsize=(8, 6))
plt.barh(
    gb_importance_df.head(10)["Variable"],
    gb_importance_df.head(10)["Importance"]
)
plt.xlabel("Importance")
plt.title("Top 10 des variables les plus importantes (Gradient Boosting)")
plt.gca().invert_yaxis()

plt.savefig("figures/gb_feature_importance.png")
plt.close()

print("\nFigure sauvegardée : figures/gb_feature_importance.png")

# =========================
# 28. Comparaison des modèles
# =========================

results = pd.DataFrame({
    "Modèle": [
        "KNN",
        "CART",
        "Random Forest",
        "Gradient Boosting"
    ],
    "Score apprentissage": [
        train_score,
        train_score_cart,
        train_score_rf,
        train_score_gb
    ],
    "Score test": [
        test_score,
        test_score_cart,
        test_score_rf,
        test_score_gb
    ]
})

print("\n===== COMPARAISON DES MODÈLES =====")
print(results)

# =========================
# 29. Figure comparaison des modèles
# =========================

plt.figure()
plt.plot(results["Modèle"], results["Score apprentissage"], marker="o", label="Apprentissage")
plt.plot(results["Modèle"], results["Score test"], marker="o", label="Test")
plt.xlabel("Modèle")
plt.ylabel("Score")
plt.title("Comparaison des performances des modèles")
plt.legend()

plt.savefig("figures/comparaison_modeles.png")
plt.close()

print("\nFigure sauvegardée : figures/comparaison_modeles.png")

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