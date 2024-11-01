import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Wczytanie danych
url = 'https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv'
data = pd.read_csv(url).dropna()

# Konwersja zmiennych kategorycznych
le = LabelEncoder()
for col in ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'region', 'income']:
    data[col] = le.fit_transform(data[col])

print("Pierwsze 5 wierszy danych:")
print(data.head())

# Informacje o danych
print("\nBrakujące wartości w danych:")
print(data.info())

# Statystyki opisowe
print("\nStatystyki opisowe")
print(data.describe())

# Sprawdzenie brakujących wartości
print("Sprawdzenie brakujących wartości")
print(data.isnull().sum())


# Usunięcie kolumny 'Unnamed: 0'
print("Usuniecie kolumny 'Unnamed: 0' jeśli istnieje...")
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# # Usuwanie kolumn z dużą ilością brakujących wartości
print("Usuwanie kolumn z dużą ilością brakujących wartości...")
missing_threshold = 0.3  # 30%
data = data.loc[:, data.isnull().mean() < missing_threshold]

# Usuwanie kolumn o niskiej wariancji
print("Usuwanie kolumn o niskiej wariancji...")
variance_threshold = 0.01
selector = VarianceThreshold(threshold=variance_threshold)
data = data[data.columns[selector.fit(data).get_support(indices=True)]]

#Usuwanie kulumny rownames

print("Usuwanie kulumny rownames...")
if 'rownames' in data.columns:
    data = data.drop('rownames', axis=1)

# # Wyodrębnienie cech i zmiennej docelowej
print("Wyodrębnienie cech i zmiennej docelowej...")
X = data.drop('score', axis=1)
y = data['score']

# Skalowanie cech
print("Skalowanie cech...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na zbiór treningowy i testowy
print("Podział danych")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)


# Definicja modeli
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# Trening i ewaluacja modeli
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"MSE": mse, "R²": r2, "MAE": mae}

print("Wyniki modeli:")
for model, metrics in results.items():
    print(f"{model}:\n MSE: {metrics['MSE']}\n R²: {metrics['R²']}\n MAE: {metrics['MAE']}\n")

# Tunowanie hiperparametrów
param_grids = {
    "Random Forest": {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10]
    },
    "XGBoost": {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [3, 5, 7]
    }
}

best_models = {}
# Grid Search z 5-krotną walidacją krzyżową
print("Grid Search z 5-krotną walidacją krzyżową")
for name, model in [("Random Forest", RandomForestRegressor(random_state=42)),
                    ("XGBoost", XGBRegressor(random_state=42))]:
    grid_search = GridSearchCV(estimator=model, param_grid=param_grids[name], cv=5,
                               scoring="neg_mean_squared_error", n_jobs=-1)

    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    print(f"Najlepsze parametry dla {name}: {grid_search.best_params_}")
    print(f"Najlepszy MSE (CV) dla {name}: {-grid_search.best_score_}\n")


# Ewaluacja najlepszych modeli na zbiorze testowym
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results[name] = {"MSE": mse, "R²": r2, "MAE": mae}


print("Wyniki najlepszych modeli po tuningu:")
for model, metrics in results.items():
    print(f"{model}:\n MSE: {metrics['MSE']}\n R²: {metrics['R²']}\n MAE: {metrics['MAE']}\n")


# Wybór kolumn numerycznych do analizy korelacji
numeric_columns = ['score', 'unemp', 'wage', 'distance', 'tuition', 'education']
data_numeric = data[numeric_columns]

# Macierz korelacji dla wybranych kolumn numerycznych
plt.figure(figsize=(8, 6))
sns.heatmap(data_numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji dla kolumn numerycznych")
plt.savefig("correlation_matrix_numeric.jpg")
plt.show()


# Macierz korelacji
plt.figure(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji")
plt.savefig("correlation_matrix.jpg")
plt.close()


# Słowniki mapujące wartości liczbowe na oryginalne kategorie
category_mappings = {
    'gender': {0: 'female', 1: 'male'},
    'ethnicity': {0: 'non-white', 1: 'white'},
    'fcollege': {0: 'no', 1: 'yes'},
    'mcollege': {0: 'no', 1: 'yes'},
    'home': {0: 'no', 1: 'yes'},
    'urban': {0: 'no', 1: 'yes'},
    'region': {0: 'south', 1: 'west', 2: 'midwest', 3: 'northeast'},
    'income': {0: 'low', 1: 'medium', 2: 'high'}
}

# Mapowanie wartości w kolumnach DataFrame na oryginalne nazwy kategorii
for col, mapping in category_mappings.items():
    if col in data.columns:
        data[col] = data[col].map(mapping)

# Tworzenie siatki 3x5 dla histogramów wszystkich kolumn
fig, axs = plt.subplots(3, 5, figsize=(20, 12))  # Ustawienia rozmiaru wykresu

# Iteracja po kolumnach i generowanie histogramów
for i, column in enumerate(data.columns):
    ax = axs[i // 5, i % 5]  # Wybór odpowiedniego miejsca w siatce 3x5
    if data[column].nunique() > 10:  # Jeśli więcej niż 10 unikalnych wartości, traktujemy jako numeryczną
        sns.histplot(data[column], kde=True, ax=ax)
    else:  # W przeciwnym razie wykres liczności dla danych kategorycznych
        sns.countplot(x=data[column], ax=ax)
    ax.set_title(column)
    ax.set_xlabel("")
    ax.set_ylabel("Liczba wystąpień")

# Dopasowanie układu i zapis do pliku
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("histograms.jpg")
plt.show()



print(f"Wykresy zostały zapisane.")
