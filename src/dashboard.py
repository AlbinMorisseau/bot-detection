import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from preprocess import preprocess_data

st.set_page_config(page_title="Explicabilité du modèle XGBoost", layout="wide")
st.title("Dashboard d'explicabilité du best modèle obten")

@st.cache_data
def load_data():
    df = pd.read_csv("../data/data.csv")
    X_train, X_test, y_train, y_test = preprocess_data(df, target_col="ROBOT")
    return X_train, X_test, y_train, y_test

@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("best_model.json")
    return model

X_train, X_test, y_train, y_test = load_data()
model = load_model()

st.success("Chargment des données et du modèle OK")


#Explicabilité avec SHAP
@st.cache_resource
def compute_shap_values(_model, X_sample):
    explainer = shap.Explainer(_model)
    shap_values = explainer(X_sample)
    return explainer, shap_values

# On prend un sous-échantillon pour accélérer le calcul SHAP
sample_size = st.slider("Taille de l'échantillon pour SHAP", 100, len(X_test), 500)
X_sample = X_test.sample(sample_size, random_state=42)

explainer, shap_values = compute_shap_values(model, X_sample)

st.subheader("Explicabilité globale")

tab1, tab2 = st.tabs(["Vue globale", "Vue locale"])

#Explicabilité globale
with tab1:
    st.write("### Importance moyenne des variables (SHAP)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_sample, plot_type="bar", show=False)
    st.pyplot(fig)

    st.write("### Détail des effets de chaque variable")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values.values, X_sample, show=False)
    st.pyplot(fig2)


#Explicabilité locale
with tab2:
    st.write("### Explication d’une prédiction spécifique")
    index = st.slider("Choisir une observation à expliquer :", 0, len(X_sample)-1, 0)
    observation = X_sample.iloc[[index]]

    st.write("#### Données de l’observation sélectionnée")
    st.dataframe(observation)

    # Force plot local
    st.write("#### Force plot (impact des features sur la prédiction)")
    shap_html = shap.getjs() + shap.force_plot(
        explainer.expected_value,
        shap_values[index].values,
        observation
    ).html()
    st.components.v1.html(shap_html, height=300)

