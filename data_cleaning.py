def check_and_clean_data(data, dataset_name="", debug=False):
    """
    Ellenőrzi a duplikátumokat és a hiányzó adatokat, majd megtisztítja az adatokat.
    :param data: A Pandas DataFrame, amelyet ellenőrizni és tisztítani kell.
    :param dataset_name: Az adathalmaz neve a naplózáshoz.
    :param debug: Ha True, részletes információkat ír ki (duplikátumok, hiányzó adatok).
    :return: A megtisztított Pandas DataFrame.
    """
    print(f"\n{dataset_name} - Ellenőrzés és tisztítás folyamatban...")
    
    # Duplikátumok ellenőrzése
    duplicates = data.duplicated()
    num_duplicates = duplicates.sum()
    if num_duplicates > 0:
        print(f"Talált duplikált sorok száma: {num_duplicates}.")
        if debug:
            print("\nDuplikált sorok:")
            print(data[duplicates])
        print("Duplikált sorok eltávolítása...")
        data = data.drop_duplicates()
    else:
        print("Nincsenek duplikált sorok.")
    
    # Hiányzó adatok ellenőrzése
    missing_values = data.isnull().sum()
    total_missing = missing_values.sum()
    if total_missing > 0:
        print(f"Hiányzó értékek találhatók: {total_missing}.")
        if debug:
            print("\nHiányzó adatokkal rendelkező sorok:")
            print(data[data.isnull().any(axis=1)])
        print("Hiányzó értékek pótlása az oszlop medián értékével...")
        data = data.fillna(data.median())
    else:
        print("Nincsenek hiányzó értékek.")
    
    print(f"{dataset_name} - Adattisztítás befejezve.")
    return data