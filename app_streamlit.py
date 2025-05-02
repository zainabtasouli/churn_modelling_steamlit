import streamlit as st
import pickle
import numpy as np

# --- Chargement du modèle et des colonnes attendues ---
@st.cache_data
def load_model():
    with open("random_model.pkl", "rb") as f:
        model = pickle.load(f)
    # Récupérer dynamiquement les noms des features
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
    else:
        st.error("Le modèle ne contient pas `feature_names_in_`. Utilise scikit-learn >=1.0.")
        st.stop()
    return model, cols

model, model_columns = load_model()

st.title("Prédiction de Churn")
st.write("Remplis les informations client pour prédire s'il va quitter.")

# --- Collecte des inputs utilisateur ---
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
age          = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure       = st.number_input("Tenure (années)", min_value=0, max_value=10, value=3)
balance      = st.number_input("Balance (€)", min_value=0.0, value=50000.0, step=100.0)
products     = st.number_input("Nombre de produits", min_value=1, max_value=4, value=2)
has_card     = st.selectbox("Possède une carte de crédit ?", ("Non", "Oui"))
is_active    = st.selectbox("Membre actif ?", ("Non", "Oui"))
salary       = st.number_input("Salaire estimé (€)", min_value=0.0, value=50000.0, step=500.0)
gender       = st.selectbox("Genre", ("Male", "Female"))
geo          = st.selectbox("Géographie", ("France", "Germany", "Spain"))

# Encoder les inputs
has_card = 1 if has_card == "Oui" else 0
is_active = 1 if is_active == "Oui" else 0
gender = 1 if gender == "Male" else 0
# dummies géo (France = base)
geo_germany = 1 if geo == "Germany" else 0
geo_spain   = 1 if geo == "Spain"   else 0

# Construire le dict complet
input_dict = {
    'CreditScore':        credit_score,
    'Gender':             gender,
    'Age':                age,
    'Tenure':             tenure,
    'Balance':            balance,
    'NumOfProducts':      products,
    'HasCrCard':          has_card,
    'IsActiveMember':     is_active,
    'EstimatedSalary':    salary,
    'Geography_Germany':  geo_germany,
    'Geography_Spain':    geo_spain
}

# Bouton de prédiction
if st.button("Prédire le churn"):
    # Construire la liste dans l'ordre
    features = [input_dict[col] for col in model_columns]
    X = np.array([features])
    # Vérification rapide
    if X.shape[1] != model.n_features_in_:
        st.error(f"Le modèle attend {model.n_features_in_} features, on en a {X.shape[1]}")
    else:
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][pred]
        label = "VA QUITTER" if pred == 1 else "VA RESTER"
        st.success(f"🔮 Le client **{label}** (confiance : {prob:.1%})")
