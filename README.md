# W związku z tym, iż self_runner odmówił współpracy na PJATK-ASI-2024/Lab-3_s24223 zostało stworzone to repozytorium.

Dokumentację modelu można znaleźć w pliku [Dokumentacja.pdf](https://github.com/s24223/Lab3_s24223/blob/main/Dokumentacja.pdf) znajdującego się w repozytorium

# Opis kroków w prediction.py

1. **Importowanie bibliotek:**
   Na początku skrypt importuje potrzebne biblioteki takie jak `pandas`, `numpy`, `scikit-learn` do obróbki danych i trenowania modeli oraz `matplotlib` do wizualizacji wyników.

2. **Pobieranie danych:**
   Dane są pobierane z pliku CSV, który może być załadowany lokalnie lub bezpośrednio z URL przy użyciu `pandas.read_csv()`. W naszym przypadku korzystamy z danych dostępnych online pod adresem [link do danych](https://vincentarelbundock.github.io/Rdatasets/csv/AER/CollegeDistance.csv).

3. **Czyszczenie danych:**
   Sprawdzane są braki danych, a następnie wypełniane są one średnimi wartościami bądź usuwane zgodnie z potrzebą

4. **Podział na zbiór treningowy i testowy:**
   Dane są dzielone na zbiór treningowy (70%) i testowy (30%) za pomocą funkcji `train_test_split` z `scikit-learn`. Zmienna zależna (czyli kolumna, którą próbujemy przewidzieć) to `score`.

5. **Trenowanie modeli:**
   W skrypcie trenowane są trzy modele:
   - **Regresja liniowa:** Prosty model przewidujący wartości `score` na podstawie relacji liniowej pomiędzy zmiennymi niezależnymi a zmienną zależną.
   - **Lasy losowe:** Model nieliniowy bazujący na drzewach decyzyjnych, używający losowych prób danych do poprawy dokładności.
   - **XGBOOST

6. **Ewaluacja modelu:**
   Każdy model jest oceniany przy użyciu trzech metryk:
   - **Mean Squared Error (MSE):** Miara średniego błędu kwadratowego.
   - **R-squared (R²):** Proporcja wyjaśnionej wariancji danych przez model.
   - **Mean Absolute Error (MAE):** Średnia wartość bezwzględnych błędów.

7. **Generowanie wykresów:**
   Wykresy przedstawiające porównanie wartości rzeczywistych i przewidzianych są tworzone przy użyciu `matplotlib`. Wykresy te są następnie zapisywane jako pliki JPG.
