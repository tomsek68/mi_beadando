from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import threading
import time
import sys


def ask_user(question, default="igen", timeout=5):
    """
    Felhasználói prompt időkorláttal és KeyboardInterrupt kezeléssel.
    :param question: A kérdés, amit felteszünk a felhasználónak.
    :param default: Az alapértelmezett válasz ("igen" vagy "nem").
    :param timeout: Időtúllépés másodpercekben.
    :return: A felhasználó válasza ("igen" vagy "nem").
    """
    user_input = []
    print(f"\n{question} (i/igen vagy n/nem, alapértelmezett: {default}, időkorlát: {timeout} másodperc)")

    def get_input():
        try:
            user_input.append(input("Válasz: ").strip().lower())
        except KeyboardInterrupt:
            print("\nMegszakítás történt. Alapértelmezett válasz kerül alkalmazásra.")
            sys.exit(0)

    # Indíts egy külön szálat az input begyűjtéséhez
    input_thread = threading.Thread(target=get_input)
    input_thread.daemon = True  # A szál megszűnik, ha a fő program kilép
    input_thread.start()

    # Várjunk a válaszra
    input_thread.join(timeout=timeout)
    if not user_input:
        print(f"\nIdőtúllépés: alapértelmezett válasz ({default}) kiválasztva.")
        return default.lower()

    # Értelmezzük a választ
    answer = user_input[0]
    return "igen" if answer in {"i", "igen"} else "nem" if answer in {"n", "nem"} else default.lower()


def feature_selection(X, y, k=5):
    """
    A legjobb k jellemző kiválasztása a SelectKBest alapján.
    :param X: A bemeneti adatok.
    :param y: A célváltozó.
    :param k: A kiválasztott jellemzők száma.
    :return: A kiválasztott jellemzőket tartalmazó adat.
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = selector.get_support(indices=True)
    print(f"Kiválasztott jellemzők indexei: {selected_features}")
    return X_selected, selected_features


def perform_grid_search(X_train, y_train):
    """
    Grid Search a legjobb hyperparaméterek megtalálásához.
    :param X_train: Tanító bemeneti adatok.
    :param y_train: Tanító címkék.
    :return: A legjobb paraméterekkel rendelkező modell.
    """
    param_grid = {
        'hidden_layer_sizes': [(100, 50), (150, 100, 50), (200, 150, 100)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'lbfgs'],
        'learning_rate_init': [0.001, 0.01, 0.0001]
    }

    grid = GridSearchCV(
        estimator=MLPClassifier(max_iter=1000, random_state=42),
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1
    )

    print("Grid Search elindult...")
    grid.fit(X_train, y_train)
    print(f"Legjobb paraméterek: {grid.best_params_}")
    print(f"Legjobb pontosság: {grid.best_score_:.2f}")
    return grid.best_estimator_


def train_neural_network(data, k_features=5):
    """
    Tréningel egy mesterséges neurális hálózatot a bor adatok alapján, Grid Search-csel és opcionális Feature Selection-nel.
    :param data: A tisztított Pandas DataFrame.
    :param k_features: A kiválasztandó jellemzők száma, ha Feature Selection-t alkalmazunk.
    :return: A betanított modell, a teszt adatok (X_test), és a teszt címkék (y_test).
    """
    print("Neural Network tréning kezdődik...")

    # Adatok szétválasztása
    X = data.drop("quality", axis=1)
    y = data["quality"]

    # Skálázás megkérdezése
    if ask_user("Szeretnéd normalizálni az adatokat (skálázás)?") == "igen":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("Adatok normalizálva.\n")

    # Feature Selection megkérdezése
    if ask_user("Szeretnéd alkalmazni a Feature Selection-t?") == "igen":
        X, selected_features = feature_selection(X, y, k=k_features)
        print(f"Kiválasztott jellemzők száma: {len(selected_features)}\n")

    # Tanító és teszt adatok felosztása
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Adatok felosztva: {len(X_train)} tanító, {len(X_test)} teszt.\n")

    # Hyperparaméterek finomhangolása Grid Search-csel
    model = perform_grid_search(X_train, y_train)

    # Modell kiértékelése
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Pontosság a teszt adatokon: {accuracy:.2f}")
    print("\nOsztályozási jelentés:")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test