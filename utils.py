import os
import zipfile
import requests
import shutil
import socket
import pandas as pd

def check_internet_connection():
    """Ellenőrzi, hogy van-e internetkapcsolat."""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        return True
    except OSError:
        return False

def get_user_choice(prompt):
    """Kérdés feltevése a felhasználónak, elfogadva a `i/igen` és `n/nem` válaszokat."""
    while True:
        choice = input(prompt).strip().lower()
        if choice in ['i', 'igen']:
            return True
        elif choice in ['n', 'nem']:
            return False
        else:
            print("Kérlek, adj meg egy érvényes választ: i/igen vagy n/nem.")

def load_config(file_path="config.csv"):
    """Konfiguráció betöltése egy CSV fájlból."""
    try:
        config = pd.read_csv(file_path)
        config_dict = dict(zip(config['key'], config['value']))
        return config_dict
    except Exception as e:
        print(f"Hiba a konfiguráció betöltésekor: {e}")
        exit(1)

def download_and_extract_wine_quality_data(config):
    """Adatok letöltése és kicsomagolása a konfiguráció alapján."""
    url = config["url"]
    project_folder = config["project_folder"]
    required_files = [config["red_wine_file"], config["white_wine_file"], config["names_file"]]

    if os.path.exists(project_folder):
        print(f"A projekt mappa ({project_folder}) már létezik.")
        existing_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(project_folder)
            for file in files
            if file in required_files
        ]
        if existing_files:
            print("Megtalált fájlok a meglévő mappában:")
            for file in existing_files:
                print(file)
        else:
            print("A meglévő mappában nem találhatók a szükséges fájlok.")
        if not check_internet_connection():
            if not existing_files:
                print("Nincs internetkapcsolat, és a szükséges fájlok hiányoznak. Kilépés.")
                exit(1)
            print("Nincs internetkapcsolat. A meglévő fájlokat fogom használni.")
            return existing_files
        if get_user_choice("Letöltsem újra az adatokat (i/igen vagy n/nem)? "):
            print("Korábbi mappa törlése és új adatok letöltése...")
            shutil.rmtree(project_folder)
            os.makedirs(project_folder, exist_ok=True)
        else:
            print("A meglévő adatok használata...")
            return existing_files if existing_files else []

    if not check_internet_connection():
        print("Nincs internetkapcsolat. Az adatok letöltése sikertelen. Kilépés.")
        exit(1)

    zip_path = os.path.join(project_folder, "wine_quality.zip")
    os.makedirs(project_folder, exist_ok=True)

    print("Letöltés folyamatban...")
    try:
        response = requests.get(url)
        with open(zip_path, "wb") as file:
            file.write(response.content)
        print("ZIP fájl letöltve.")
    except Exception as e:
        print(f"Letöltési hiba: {e}. Kilépés.")
        exit(1)

    print("Kicsomagolás...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(project_folder)
    print("Kicsomagolás kész.")

    target_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(project_folder)
        for file in files
        if file in required_files
    ]

    if target_files:
        print("Megtalált fájlok:")
        for file in target_files:
            print(file)
    else:
        print("A keresett fájlok nem találhatók a letöltött adatokban sem.")
    return target_files

if __name__ == "__main__":
    config = load_config()
    print("Bor minőség előrejelzés projekt inicializálása...")
    found_files = download_and_extract_wine_quality_data(config)
    if found_files:
        print("\nSikeresen előkészítettük a projektet!")
    else:
        print("\nHiba történt az előkészítés során!")