import os
import socket
import subprocess
import streamlit as st
import pandas as pd
from stlit_plot_definitions import (
    load_config,
    load_data,
    plot_histogram,
    plot_boxplot,
    plot_scatter,
    plot_correlation_matrix,
    display_confusion_matrix,
    prepare_data,
    gather_inputs,
    load_model
)


def is_streamlit_running(host="localhost", port=8501):
    """
    Ellenőrzi, hogy a Streamlit szerver fut-e a megadott hoston és porton.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)  # Maximum várakozási idő
        try:
            sock.connect((host, port))
            return True
        except (socket.timeout, ConnectionRefusedError):
            return False


def main():
    st.title("Bor minőség vizualizáció és predikció")

    # Konfiguráció betöltése
    config = load_config()
    if not config:
        return

    # Adatok betöltése
    red_wine_data = load_data(config, "red_wine_file")
    white_wine_data = load_data(config, "white_wine_file")

    if red_wine_data is None or white_wine_data is None:
        return

    # Bor típus választása
    wine_type = st.selectbox("Válassz bortípust", ["Vörösbor", "Fehérbor"], key="wine_type_selector")

    # Megfelelő adat kiválasztása és modell fájl neve
    if wine_type == "Vörösbor":
        wine_data = red_wine_data
        model_filename = "red_wine_model.pkl"
        dataset_name = "Vörösbor"
    else:
        wine_data = white_wine_data
        model_filename = "white_wine_model.pkl"
        dataset_name = "Fehérbor"

    # Modell és jellemzők betöltése
    model, selected_features = load_model(model_filename)
    if model is None:
        st.error("Nem sikerült betölteni a modellt.")
        return

    # Ha nincsenek mentett jellemzők, használjuk az adatok oszlopait
    if selected_features is None:
        selected_features = list(wine_data.columns[:-1])  # Az összes oszlop, kivéve a 'quality'

    # Adatok szűrése a modell jellemzőire
    try:
        wine_data_filtered = prepare_data(wine_data, selected_features + ["quality"])
    except KeyError as e:
        st.error(f"Hiba: Az adathalmaz nem tartalmazza a szükséges jellemzőket: {e}")
        return

    # Fülek létrehozása
    tabs = st.tabs(["CSV Megjelenítő", "Vizualizáció", "Konfúziós Mátrix", "Bor Predikció"])

    # CSV megjelenítő fül
    with tabs[0]:
        st.subheader(f"{dataset_name} - CSV Megjelenítő")
        st.dataframe(wine_data_filtered.head())

    # Vizualizáció fül
    with tabs[1]:
        st.subheader(f"{dataset_name} - Vizualizációk")
        visualization_type = st.selectbox(
            "Válassz diagram típust",
            ["Hisztogram", "Boxplot", "Szórásdiagram", "Korrelációs mátrix"],
            key="visualization_type_selector"
        )
        if visualization_type == "Hisztogram":
            column = st.selectbox("Válassz oszlopot", wine_data_filtered.columns, key="hist_column")
            plot_histogram(wine_data_filtered, column)
        elif visualization_type == "Boxplot":
            column = st.selectbox("Válassz oszlopot", wine_data_filtered.columns, key="box_column")
            plot_boxplot(wine_data_filtered, column)
        elif visualization_type == "Szórásdiagram":
            x_column = st.selectbox("X tengely oszlop", wine_data_filtered.columns, key="scatter_x_column")
            y_column = st.selectbox("Y tengely oszlop", wine_data_filtered.columns, key="scatter_y_column")
            plot_scatter(wine_data_filtered, x_column, y_column)
        elif visualization_type == "Korrelációs mátrix":
            plot_correlation_matrix(wine_data_filtered)

    # Konfúziós Mátrix fül
    with tabs[2]:
        st.subheader("Konfúziós Mátrix - Neural Network kiértékelés")
        display_confusion_matrix(wine_data_filtered, model)

    # Bor Predikció fül
    with tabs[3]:
        st.subheader(f"{dataset_name} - Bor Minőség Predikció")

        # Felhasználó által megadható jellemzők
        input_data = gather_inputs(selected_features, wine_data_filtered)
        prediction = model.predict(input_data)[0]
        st.success(f"A modell előrejelzése a bor minőségére: **{prediction}**")


if __name__ == "__main__":
    if is_streamlit_running():
        print("A Streamlit szerver már fut. Nyissa meg a böngészőben: http://localhost:8501")
        main()
    else:
        print("A Streamlit szerver elindítása...")
        subprocess.run(["streamlit", "run", "visualizer_streamlit.py"], check=True)