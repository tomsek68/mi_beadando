import os
import subprocess
from utils import download_and_extract_wine_quality_data, load_config
from data_cleaning import check_and_clean_data
from model_training import train_neural_network

import pandas as pd
import pickle


def clear_terminal():
    """Terminál tisztítása."""
    os.system('cls' if os.name == 'nt' else 'clear')


def load_data(files):
    """
    Betölti a vörösbor és fehérbor adatait a megadott fájlokból.
    :param files: A fájlok listája.
    :return: Vörösbor és fehérbor adatai DataFrame-ben.
    """
    red_wine_data = None
    white_wine_data = None
    for file in files:
        if "winequality-red.csv" in file:
            red_wine_data = pd.read_csv(file, sep=';')
        elif "winequality-white.csv" in file:
            white_wine_data = pd.read_csv(file, sep=';')
    return red_wine_data, white_wine_data


def save_model(model, filename):
    """Modell mentése pickle fájlba."""
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def main():
    clear_terminal()
    print("Bor minőség előrejelzés projekt inicializálása...")

    # Konfiguráció betöltése
    config = load_config()

    # Adatok letöltése és kicsomagolása
    found_files = download_and_extract_wine_quality_data(config)
    if not found_files:
        print("Hiba történt az adatok előkészítése során!")
        return

    # Adatok betöltése
    red_wine_data, white_wine_data = load_data(found_files)

    # Adattisztítás
    if red_wine_data is not None:
        red_wine_data = check_and_clean_data(red_wine_data, dataset_name="Vörösbor")
    if white_wine_data is not None:
        white_wine_data = check_and_clean_data(white_wine_data, dataset_name="Fehérbor")

    # Neural Network tréning és modellek mentése
    if red_wine_data is not None:
        print("\nVörösbor modell tréning:")
        red_wine_model, X_test, y_test = train_neural_network(red_wine_data)
        save_model(red_wine_model, "red_wine_model.pkl")

    if white_wine_data is not None:
        print("\nFehérbor modell tréning:")
        white_wine_model, X_test, y_test = train_neural_network(white_wine_data)
        save_model(white_wine_model, "white_wine_model.pkl")

    # Streamlit alkalmazás indítása a vizualizációhoz
    print("\nIndítom a vizualizációs alkalmazást...")
    subprocess.run(["streamlit", "run", "visualizer_streamlit.py"])


if __name__ == "__main__":
    main()