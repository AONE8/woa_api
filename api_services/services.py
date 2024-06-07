import pandas as pd
import numpy as np
from mealpy import FloatVar
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from mealpy.swarm_based.WOA import OriginalWOA


def woa_selection_service(data_csv_url, whales_num=50, iters_num=30):
    # Завантаження даних
    data = pd.read_csv(data_csv_url)

    # Розділення даних на ознаки та залежну змінну
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Вибір числових стовпців
    numeric_columns = X.select_dtypes(include=[np.number]).columns

    # Ініціалізація DataFrame для імпутованих даних
    data_imputed = pd.DataFrame()

    # Заміна пропущених значень на середнє
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    data_imputed[numeric_columns] = pd.DataFrame(imputer.fit_transform(data[numeric_columns]), columns=numeric_columns)

    # Вибір стовпців з категоріями
    categorical_columns = X.select_dtypes(include=['object']).columns

    # Додавання категоріальних стовпців назад до датасету без змін
    data_imputed[categorical_columns] = X[categorical_columns]

    # Набір ознак для відповіді
    X_response = data_imputed

    # Кодування категоріальних даних
    categorical_data = data_imputed[categorical_columns]
    one_hot_encoder = OneHotEncoder()
    categorical_encoded = one_hot_encoder.fit_transform(categorical_data).toarray()

    # Видалення оригінальних категоріальних стовпців
    data_imputed.drop(categorical_columns, axis=1, inplace=True)

    # Додавання закодованих категоріальних даних до датасету
    data_imputed = np.hstack((data_imputed.values, categorical_encoded))

    print(f"data_imputed={data_imputed}")

    # Кодування залежну зміну

    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = y.values

    print(f"y_encoded={y_encoded}")

    # Підготовка даних для вибору ознак за допомогою WOA
    X_prepared = data_imputed
    y_prepared = y_encoded

    # Функція пристосованості
    def objective_function(solution):
        # Обмеження рішення до 0 або 1
        binary_solution = np.where(solution > 0.5, 1, 0)

        # Перевірка чи є хоча б одна ознака вибрана
        if np.sum(binary_solution) == 0:
            # Повертаємо нескінченність, якщо не вибрано жодної ознаки
            return float('inf')

        # Розділення даних на навчальні та валідаційні набори
        X_train, X_val, y_train, y_val = train_test_split(
            X_prepared[:, binary_solution == 1], y_prepared, test_size=0.2, random_state=0
        )

        # Створення та навчання моделі Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        # Оцінка моделі на валідаційних даних
        predictions = model.predict(X_val)
        mse = mean_squared_error(y_val, predictions)

        return mse

    # Ініціалізація WOA
    model = OriginalWOA(epoch=iters_num, pop_size=whales_num)

    # Створення словника з параметрами проблеми
    problem_dict = {
        "bounds": FloatVar(lb=(-1,) * X_prepared.shape[1], ub=(1,) * X_prepared.shape[1], name="agro_data"),
        "minmax": "min",
        "obj_func": objective_function
    }

    # Запуск оптимізації
    g_best = model.solve(problem_dict)

    best_solution = g_best.solution
    best_fitness = g_best.target.fitness

    # Визначення індексів вибраних ознак
    selected_features = np.where(best_solution > 0.5)[0]

    # Визначення стовпців вибраних ознак
    selected_features_columns = data.columns[selected_features]

    # Виведення результату
    response = {
        "best_solution": best_solution.tolist(),
        "best_fitness": best_fitness,
        "selected_features_columns": selected_features_columns.tolist(),
        "X": X_response[selected_features_columns].to_dict(orient='records'),
        "y": pd.DataFrame(y).to_dict(orient='records')
    }

    return response
