import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from fpdf import FPDF

from sklearn.feature_selection import VarianceThreshold
# Biblioteki do modelowania
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# Wczytanie danych z podanego URL
url = 'https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv'
data = pd.read_csv(url)
data = data.dropna()
#zmiana str na double
le = LabelEncoder()
for col in ['gender', 'ethnicity', 'fcollege', 'mcollege', 'home', 'urban', 'region', 'education', 'income']:
    data[col] = le.fit_transform(data[col])

# Wyświetlenie pierwszych 5 wierszy
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



# Usunięcie kolumny 'Unnamed: 0' jeśli istnieje
print("Usuniecie kolumny 'Unnamed: 0' jeśli istnieje...")
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)

# Usuwanie kolumn z dużą ilością brakujących wartości
print("Usuwanie kolumn z dużą ilością brakujących wartości")
missing_threshold = 0.3  # 30%
data = data.loc[:, data.isnull().mean() < missing_threshold]

# Usuwanie kolumn o niskiej wariancji
variance_threshold = 0.01  # Próg wariancji
selector = VarianceThreshold(threshold=variance_threshold)
data_reduced = data[data.columns[selector.fit(data).get_support(indices=True)]]

# usunięcie danych o małej korelacji
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
threshold = 0.1  # Ustal próg korelacji
low_corr_cols = correlation_matrix[abs(correlation_matrix['score']) < threshold].index.tolist()

# Sprawdzenie zawartości listy low_corr_cols przed usunięciem
print("Kolumny o niskiej korelacji z 'score':", low_corr_cols)

# Usunięcie 'score' z listy, jeśli jest w low_corr_cols
if 'score' in low_corr_cols:
    low_corr_cols.remove('score')  # Usuń 'score' z listy

# Sprawdzenie zmiennych kategorycznych
print ("Sprawdzenie zmiennych kategorycznych")
print(data.select_dtypes(include=['object']).columns)

# Kodowanie zmiennych kategorycznych
print("Kodowanie zmiennych kategorycznych")
data_encoded = pd.get_dummies(data, drop_first=True)

# Wyodrębnienie cech i zmiennej docelowej
print("Wyodrębnienie cech i zmiennej docelowej")
X = data_encoded.drop('score', axis=1)
y = data_encoded['score']

# Skalowanie cech
print("Skalowanie cech")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział danych
print("Podział danych")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# Trening modeli i ocena

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Wyświetlenie współczynników regresji
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print("\n")

    results[name] = {"MSE": mse, "R²": r2, "MAE": mae}

print("Wyniki modeli:")
for model, metrics in results.items():
    print(f"{model}:\n MSE: {metrics['MSE']}\n R²: {metrics['R²']}\n MAE: {metrics['MAE']}\n")



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

# Grid Search z 5-krotną walidacją krzyżową
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name} - Mean CV MSE: {-scores.mean()}")

# Zdefiniowanie słownika do przechowywania najlepszych modeli
best_models = {}

# Iteracja przez modele i ich parametry do tuningu
for name, model in [("Random Forest", RandomForestRegressor(random_state=42)),
                    ("XGBoost", XGBRegressor(random_state=42))]:
    print(f"Tuning hiperparametrów dla {name}...")

    # Utworzenie obiektu GridSearchCV
    grid_search = GridSearchCV(estimator=model,param_grid=param_grids[name], cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Zapisanie najlepszego modelu
    best_models[name] = grid_search.best_estimator_

    print(f"Najlepsze parametry dla {name}: {grid_search.best_params_}")
    print(f"Najlepszy MSE (CV) dla {name}: {-grid_search.best_score_}\n")

# Ewaluacja najlepszych modeli na zbiorze testowym
results = {}
for name, model in best_models.items():
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    results[name] = {"MSE": mse, "R²": r2, "MAE": mae}

# Wyświetlenie wyników
print("Wyniki najlepszych modeli po tuningu:")
for model, metrics in results.items():
    print(f"{model}:\n MSE: {metrics['MSE']}\n R²: {metrics['R²']}\n MAE: {metrics['MAE']}\n")



# Tworzenie dokumentu Readme.pdf
pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "B", 16)
pdf.cell(200, 10, "Dokumentacja modelu predykcyjnego", new_x="LMARGIN", new_y="NEXT", align="C")

# Wyznaczenie najlepszego modelu na podstawie najniższego MSE
best_model_name = min(results, key=lambda x: results[x]["MSE"])
best_model_metrics = results[best_model_name]

# Wyniki najlepszego modelu
pdf.set_font("Helvetica", "B", 12)
pdf.cell(200,10,f"Wybrany model: {best_model_name}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(200, 10, "\nModel wybrane ze wzgledu na najlepsze parametry tj:", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
pdf.cell(200, 10, f"Model: {best_model_name}\n", new_x="LMARGIN", new_y="NEXT")
pdf.cell(200, 10, f"MSE: {best_model_metrics['MSE']}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(200, 10, f"R²: {best_model_metrics['R²']}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(200, 10, f"MAE: {best_model_metrics['MAE']}", new_x="LMARGIN", new_y="NEXT")


# Pierwsze 5 wierszy danych
pdf.set_font("Helvetica", "B", 12)
pdf.cell(200, 10, "Pierwsze 5 wierszy danych:", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)
for i, row in data.head(5).iterrows():
    pdf.cell(200, 10, str(row.values).encode('latin1', 'replace').decode('latin1'), new_x="LMARGIN", new_y="NEXT")


# Statystyki opisowe
stats = data.describe()

# Dodawanie tabeli statystyk do PDF
pdf.set_font("Helvetica", "B", 12)
pdf.cell(200, 10, "\nStatystyki:", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 10)


# Dodawanie nagłówków tabeli
header = ["Statystyka"] + list(stats.columns)
for column in header:
    pdf.cell(30, 10, column, 1)
pdf.ln()

# Dodawanie wierszy ze statystykami
for stat_name, row in stats.iterrows():
    pdf.cell(30, 10, stat_name, 1)
    for value in row:
        pdf.cell(30, 10, str(round(value,3)), 1)  # Zaokrąglanie wartości do 2 miejsc po przecinku
    pdf.ln()


# Macierz korelacji, zapisanie jako obraz
plt.figure(figsize=(10, 8))
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".3f", cmap="coolwarm")
plt.title('Macierz korelacji')
plt.savefig('correlation_matrix.jpg', format='jpg')
plt.close()
print(f'Zapisano macierz korelacji.')


# Rozkład zmiennej docelowej 'score'
plt.figure()
sns.histplot(data['score'], kde=True)
plt.title('Rozkład zmiennej score')
plt.xlabel('Score')
plt.ylabel('Liczność')
plt.savefig('rozklad_score.jpg', format='jpg')
plt.close()

# Wykres reszt
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals)
plt.hlines(y=0, xmin=min(y_pred), xmax=max(y_pred), colors='red')
plt.xlabel('Przewidywane wartości')
plt.ylabel('Reszty')
plt.title('Wykres reszt')
plt.savefig('wykres_reszt.jpg', format='jpg')
plt.close()


# Generowanie histogramow dla wszystkich danych z conajmniej 3 roznymi wartościami nie licząc score dla ktorego został wczesniej wykonany wykres
existing_columns_to_plot = existing_columns_to_plot = [col for col in data.columns if data[col].nunique() > 2 or col != "score"]
for column in existing_columns_to_plot:
    plt.figure()
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram kolumny {column}')
    plt.xlabel(column)
    plt.ylabel('Liczba wystąpień')
    plt.savefig(f'histogram_{column}.jpg', format='jpg')
    plt.close()
    print(f'Zapisano histogram dla kolumny: {column}')



#Zapis wszyskich wykresów do pdf
pdf.cell(200, 10, "\nWykresy:", new_x="LMARGIN", new_y="NEXT")
pdf.image("correlation_matrix.jpg", x=10, w=180)
pdf.image("rozklad_score.jpg", x=10, w=180)
pdf.image("wykres_reszt.jpg", x=10, w=180)
for column in existing_columns_to_plot:
    filename = f'histogram_{column}.jpg'
    if os.path.exists(filename):
        pdf.image(filename, x=10, w=180)
    else:
        print(f'Plik {filename} nie istnieje.')

pdf.cell(200,10,"\n\nKoniec dokumentacji")
pdf.output("Readme.pdf")
