from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample
import pandas as pd
from plyer import notification
import threading
import queue
import time


def ask_user_popup(question, default="igen", timeout=10):
    """
    Felhasználói kérdés felugró ablakon és terminálon keresztül.
    """
    response_queue = queue.Queue()

    # Értesítés megjelenítése
    notification.notify(
        title="Döntés szükséges",
        message=f"{question}\nVálaszolj terminálon! (alapértelmezett: {default})",
        timeout=5
    )

    def user_input():
        try:
            user_input = input(f"{question} (i/igen vagy n/nem, alapértelmezett: {default}): ").strip().lower()
            response_queue.put(user_input)
        except KeyboardInterrupt:
            response_queue.put(None)

    # Input szál indítása
    input_thread = threading.Thread(target=user_input)
    input_thread.start()

    # Várakozás időkorlátig
    input_thread.join(timeout)

    if not response_queue.empty():
        user_response = response_queue.get()
        if user_response in {"i", "igen"}:
            return "igen"
        elif user_response in {"n", "nem"}:
            return "nem"

    # Ha időtúllépés vagy üres válasz
    print(f"Időtúllépés vagy üres válasz: alapértelmezett ({default}) alkalmazva.")
    return default.lower()


def feature_selection(X, y, k=5):
    """
    A legjobb k jellemző kiválasztása a SelectKBest alapján.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    print(f"Kiválasztott jellemzők indexei: {selected_features}")
    return X_selected, selected_features


def oversample_classes(data, target_column="quality"):
    """
    Túlsúlyozás: kiegyensúlyozza az osztályokat az összes osztályt a legnagyobb osztály méretéhez igazítva.
    """
    max_class_count = data[target_column].value_counts().max()
    
    # Minden osztályt túlsúlyozunk
    oversampled_data = pd.concat([
        resample(data[data[target_column] == value], 
                 replace=True,  # Újramintavétel engedélyezése
                 n_samples=max_class_count,  # A legnagyobb osztály méretéhez igazítjuk
                 random_state=42)
        for value in data[target_column].unique()
    ])
    
    return oversampled_data


def perform_grid_search(X_train, y_train, X_test, y_test):
    """
    Grid Search a legjobb hyperparaméterek megtalálásához és osztályozási jelentés kiírása.
    """
    param_grid = {
        'hidden_layer_sizes': [(20, 10, 1), (50, 30, 10)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'learning_rate_init': [0.001, 0.01, 0.0001, 0.1]
    }

    grid = GridSearchCV(
        estimator=MLPClassifier(max_iter=10000, random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )

    print("Grid Search elindult...")
    grid.fit(X_train, y_train)

    # Legjobb paraméterek és pontosság kiírása
    print(f"Legjobb paraméterek: {grid.best_params_}")
    print(f"Legjobb pontosság a validációs adatokon: {grid.best_score_:.2f}")

    # Legjobb modell kiértékelése a teszt adatokon
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Pontosság a teszt adatokon (legjobb modell): {accuracy:.2f}")
    print("\nOsztályozási jelentés (legjobb modell):")
    print(classification_report(y_test, y_pred))

    return best_model


def log_quality_distribution(y, title):
    """
    Kiírja az osztályok előfordulási gyakoriságát a quality változóban.
    """
    print(f"\n{title}:")
    counts = y.value_counts().sort_index()
    for quality, count in counts.items():
        print(f"Quality {quality}: {count} példány")
    print()


def train_neural_network(data, k_features=5):
    """
    Tréningel egy mesterséges neurális hálózatot a bor adatok alapján, Grid Search-csel és opcionális Feature Selection-nel.
    """
    print("Neural Network tréning kezdődik...")

    # Adatok szétválasztása
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # Minőségeloszlás megjelenítése
    log_quality_distribution(y, "Eredeti quality eloszlás")

    # Skálázás kiválasztása
    normalize_data = ask_user_popup("Szeretnéd normalizálni az adatokat (skálázás)?")
    if normalize_data == "igen":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Adatok normalizálva.\n")

    # Oversampling kiválasztása
    oversample_data = ask_user_popup("Szeretnéd kiegyensúlyozni az osztályokat túlsúlyozással?")
    if oversample_data == "igen":
        data = oversample_classes(data, target_column="quality")
        X = data.drop("quality", axis=1)
        y = data["quality"]
        print("Osztályok kiegyensúlyozva (oversampling).\n")

    # Feature Selection kiválasztása
    use_feature_selection = ask_user_popup("Szeretnéd alkalmazni a Feature Selection-t?")
    if use_feature_selection == "igen":
        X, selected_features = feature_selection(X, y, k=k_features)
        print(f"Kiválasztott jellemzők száma: {len(selected_features)}\n")

    # Tanító és teszt adatok felosztása
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Adatok felosztva: {len(X_train)} tanító, {len(X_test)} teszt.\n")

    # Kiírás: Teszt quality eloszlás
    log_quality_distribution(pd.Series(y_train), "Train quality eloszlás")
    log_quality_distribution(pd.Series(y_test), "Test quality eloszlás")

    # Hyperparaméterek finomhangolása és osztályozási jelentés
    model = perform_grid_search(X_train, y_train, X_test, y_test)

    return model, X_test, y_test


# Futtatás
if __name__ == "__main__":
    data_file = input("Adja meg az adatfájlt (CSV): ").strip()
    data = pd.read_csv(data_file)
    train_neural_network(data)