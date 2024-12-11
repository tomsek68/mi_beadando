import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import pickle


def load_config(file_path="config.csv"):
    """
    Konfiguráció betöltése a gyökérkönyvtárból.
    """
    try:
        abs_path = os.path.join(os.path.dirname(__file__), file_path)
        config = pd.read_csv(abs_path).set_index('key').to_dict()['value']
        return config
    except Exception as e:
        st.error(f"Nem sikerült betölteni a konfigurációt: {e}")
        return {}


def load_data(config, file_key, sep=";"):
    """
    Adatok betöltése a CSV fájlból a gyökérkönyvtár és a project_folder alapján.
    """
    try:
        project_folder = config.get("project_folder")
        file_path = os.path.join(os.path.dirname(__file__), project_folder, config[file_key])
        return pd.read_csv(file_path, sep=sep)
    except Exception as e:
        st.error(f"Nem sikerült betölteni az adatokat: {e}")
        return None


def plot_histogram(data, column):
    fig, ax = plt.subplots()
    ax.hist(data[column], bins=20, alpha=0.7, color='blue')
    ax.set_title(f'{column} eloszlása')
    ax.set_xlabel(column)
    ax.set_ylabel('Gyakoriság')
    st.pyplot(fig)


def plot_boxplot(data, column):
    fig, ax = plt.subplots()
    ax.boxplot(data[column], vert=False)
    ax.set_title(f'{column} eloszlása (boxplot)')
    ax.set_xlabel(column)
    st.pyplot(fig)


def plot_scatter(data, x_column, y_column):
    fig, ax = plt.subplots()
    ax.scatter(data[x_column], data[y_column], alpha=0.5)
    ax.set_title(f'{x_column} és {y_column} kapcsolata')
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    st.pyplot(fig)


def plot_correlation_matrix(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(data.corr(), cmap='coolwarm')
    fig.colorbar(cax)
    ax.set_xticks(range(len(data.columns)))
    ax.set_yticks(range(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=90)
    ax.set_yticklabels(data.columns)
    ax.set_title('Korrelációs mátrix', pad=20)
    st.pyplot(fig)


def load_model(filename):
    """
    Modell és (opcionálisan) jellemzők listájának betöltése pickle fájlból.
    Ha a jellemzők listája nem érhető el, használjuk az adat oszlopait.
    """
    try:
        with open(filename, "rb") as file:
            model_data = pickle.load(file)

        # Ellenőrizzük, hogy a jellemzők listája elérhető-e
        if isinstance(model_data, tuple) and len(model_data) == 2:
            model, selected_features = model_data
        else:
            model = model_data
            selected_features = None  # Ha nincsenek mentett jellemzők

        return model, selected_features

    except FileNotFoundError:
        st.error(f"A modell fájl nem található: {filename}")
        return None, None
    except Exception as e:
        st.error(f"Hiba a modell betöltésekor: {e}")
        return None, None


def prepare_data(data, selected_features):
    """
    Adatok szűrése a modell által elvárt jellemzőkre.
    """
    try:
        # Ellenőrizzük, hogy a jellemzők száma egyezik-e
        available_features = list(data.columns)
        missing_features = [feature for feature in selected_features if feature not in available_features]
        
        if missing_features:
            raise KeyError(f"Hiányzó jellemzők: {missing_features}")

        # Csak a szükséges jellemzőket hagyjuk meg
        return data[selected_features]
    except KeyError as e:
        st.error(f"Hiányzó jellemző az adatokban: {e}")
        raise


def display_confusion_matrix(data, model):
    """
    Konfúziós mátrix és osztályozási jelentés megjelenítése.
    """
    X = data.drop("quality", axis=1)
    y = data["quality"]

    if X.shape[1] != model.n_features_in_:
        st.error(f"Az adatok jellemzőinek száma ({X.shape[1]}) nem egyezik a modell elvárásaival ({model.n_features_in_}).")
        return

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    st.write(f"Pontosság az adatokon: **{accuracy:.2f}**")
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(y, y_pred, ax=ax, cmap='Blues')
    st.pyplot(fig)
    st.text(classification_report(y, y_pred, zero_division=0))


def gather_inputs(selected_features, data):
    """
    Csúszkák generálása felhasználói bemeneti értékekhez.
    Az alapértelmezett értékek az adatok átlagai.
    """
    inputs = {}
    for feature in selected_features:
        avg_value = data[feature].mean()  # Az adott oszlop átlaga
        max_value = data[feature].max() * 1.5  # Max határérték (biztonsági tartomány)
        min_value = data[feature].min() * 0.5  # Min határérték (biztonsági tartomány)
        inputs[feature] = st.slider(
            f"{feature.capitalize()}",
            min_value=float(min_value),
            max_value=float(max_value),
            value=float(avg_value),  # Alapértelmezett érték az átlag
            step=0.01,
            key=f"slider_{feature}"
        )
    return pd.DataFrame([inputs])