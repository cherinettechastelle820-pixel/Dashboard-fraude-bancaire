import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Titre du Dashboard
# =========================
st.title("Dashboard interactif – Détection de fraude bancaire 💳")

# =========================
# Chargement des données
# =========================
df = pd.read_csv("fraude_bancaire.csv")

# =========================
# Aperçu des données
# =========================
st.subheader("Aperçu du jeu de données")
st.dataframe(df.head())

# =========================
# Colonnes numériques et catégorielles
# =========================
num_cols = df.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df.select_dtypes(include=["object", "category"]).columns

st.subheader("Paramètres du graphique")

# Sélection des axes X et Y
x_axis = st.selectbox("Choisir l'axe X", num_cols)
y_axis = st.selectbox("Choisir l'axe Y", num_cols)

# Sélection de la couleur
color_var = st.selectbox("Choisir la couleur des points (optionnel)", [None] + list(cat_cols))

# Sélection de la taille des points
size_var = st.selectbox("Choisir la taille des points (optionnel)", [None] + list(num_cols))

# =========================
# Création du graphique
# =========================
fig, ax = plt.subplots(figsize=(8,6))
sns.scatterplot(
    data=df,
    x=x_axis,
    y=y_axis,
    hue=color_var,
    size=size_var,
    palette="Set2",
    sizes=(20, 200),
    alpha=0.7,
    ax=ax
)

ax.set_title(f"{x_axis} vs {y_axis}", fontsize=14)
ax.legend(loc='best', fontsize=9)
st.pyplot(fig)