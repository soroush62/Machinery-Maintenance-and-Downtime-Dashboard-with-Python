import lightningchart as lc
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc

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
    "Logistic Regression": LogisticRegression(max_iter=10000),
    "Random Forest": RandomForestClassifier(n_jobs=-1),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dashboard = lc.Dashboard(
    columns=3,
    rows=2,
    theme=lc.Themes.Dark
)

for i, (model_name, model) in enumerate(models.items()):
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    pipeline.fit(X_train, y_train)

    y_scores = pipeline.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)

    thresholds = np.nan_to_num(thresholds, nan=0.0)
    normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())

    chart = dashboard.ChartXY(
        column_index=i % 3,
        row_index=i // 3,
        row_span=1,
        column_span=1
    )
    chart.set_title(f'{model_name} Precision-Recall Curve with Thresholds (AUC = {pr_auc:.2f})')

    pr_series = chart.add_line_series()
    pr_series.add(recall.tolist(), precision.tolist()).set_name('Precision-Recall Curve')

    point_series = chart.add_point_series().set_name('Threshold Points')

    for j in range(len(thresholds)):
        color = lc.Color(
            int(255 * (1 - normalized_thresholds[j])),  # Red component decreases with threshold
            int(255 * normalized_thresholds[j]),        # Green component increases with threshold
            0                                           # Blue component stays constant
        )
        point_series.add(recall[j], precision[j]).set_point_color(color)

    chart.get_default_x_axis().set_title('Recall')
    chart.get_default_y_axis().set_title('Precision')

voting_clf = VotingClassifier(estimators=[
    ('lr', LogisticRegression(max_iter=10000)),
    ('rf', RandomForestClassifier()),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
    ('lgbm', LGBMClassifier()),
    ('cat', CatBoostClassifier(verbose=0))
], voting='soft')

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', voting_clf)
])

pipeline.fit(X_train, y_train)

y_scores = pipeline.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

thresholds = np.nan_to_num(thresholds, nan=0.0)
normalized_thresholds = (thresholds - thresholds.min()) / (thresholds.max() - thresholds.min())


chart = dashboard.ChartXY(
    column_index=2,
    row_index=1,
    row_span=1,
    column_span=1
)
chart.set_title(f'Ensemble Methods Precision-Recall Curve with Thresholds (AUC = {pr_auc:.2f})')

pr_series = chart.add_line_series()
pr_series.add(recall.tolist(), precision.tolist()).set_name('Precision-Recall Curve')

point_series = chart.add_point_series().set_name('Threshold Points')

for i in range(len(thresholds)):
    color = lc.Color(
        int(255 * (1 - normalized_thresholds[i])),  # Red component decreases with threshold
        int(255 * normalized_thresholds[i]),        # Green component increases with threshold
        0                                           # Blue component stays constant
    )
    point_series.add(recall[i], precision[i]).set_point_color(color)

chart.get_default_x_axis().set_title('Recall')
chart.get_default_y_axis().set_title('Precision')
legend = chart.add_legend(horizontal=False)
legend.add(pr_series)
legend.add(point_series)

dashboard.open()





