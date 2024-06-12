import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

lc.set_license('my-license-key')

file_path = 'X_train.csv'
data = pd.read_csv(file_path)

selected_features = ['Air_temperature_K', 'Process_temperature_K', 'Rotational_speed_rpm', 'Torque_Nm', 'Tool_wear_min']

X = data[selected_features]
y = data['Machine_failure']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), selected_features)
    ])

models = {
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'LightGBM': LGBMClassifier(),
    'CatBoost': CatBoostClassifier(verbose=0)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

features_to_plot = selected_features

dashboard = lc.Dashboard(
    rows=len(features_to_plot),
    columns=len(models),
    theme=lc.Themes.Dark
)

def plot_actual_vs_predicted(chart, feature, model_name, y_test, y_pred):
    actual_failures = X_test[y_test == 1][feature]
    predicted_failures = X_test[y_pred == 1][feature]
    
    actual_series = chart.add_point_series()
    actual_series.add(X_test[y_test == 1].index.tolist(), actual_failures.tolist())
    actual_series.set_point_color(lc.Color(255, 0, 0))
    actual_series.set_name('Actual Failures')
    
    predicted_series = chart.add_point_series()
    predicted_series.add(X_test[y_pred == 1].index.tolist(), predicted_failures.tolist())
    predicted_series.set_point_color(lc.Color(0, 0, 255))
    predicted_series.set_name('Predicted Failures')

    chart.set_title(f'{model_name} ({feature})')
    chart.get_default_x_axis().set_title('Index')
    chart.get_default_y_axis().set_title(feature)
    chart.add_legend(data=chart, horizontal=True)

for j, (model_name, model) in enumerate(models.items()):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    for i, feature in enumerate(features_to_plot):
        chart = dashboard.ChartXY(
            column_index=j,
            row_index=i,
            row_span=1,
            column_span=1
        )
        plot_actual_vs_predicted(chart, feature, model_name, y_test, y_pred)

dashboard.open()










