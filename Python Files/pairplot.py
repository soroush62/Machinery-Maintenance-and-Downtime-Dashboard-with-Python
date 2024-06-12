import lightningchart as lc
import pandas as pd
import itertools

lc.set_license('my-license-key')

file_path = 'X_train.csv'
data = pd.read_csv(file_path)

columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

dashboard = lc.Dashboard(
    rows=len(columns),
    columns=len(columns),
    theme=lc.Themes.Dark
)

def create_scatter_chart(dashboard, title, x_values, y_values, xlabel, ylabel, column_index, row_index):
    chart = dashboard.ChartXY(
        column_index=column_index,
        row_index=row_index
    )
    chart.set_title(title)

    scatter_series = chart.add_point_series()
    scatter_series.add(x_values, y_values)

    chart.get_default_x_axis().set_title(xlabel)
    chart.get_default_y_axis().set_title(ylabel)

for row_index, y_col in enumerate(columns):
    for column_index, x_col in enumerate(columns):
        x_values = data[x_col].astype(float).tolist()
        y_values = data[y_col].astype(float).tolist()
        title = f'{x_col} vs {y_col}'
        create_scatter_chart(dashboard, title, x_values, y_values, x_col, y_col, column_index, row_index)

dashboard.open()



