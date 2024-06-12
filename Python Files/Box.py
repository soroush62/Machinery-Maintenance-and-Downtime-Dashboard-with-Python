import lightningchart as lc
import pandas as pd
import numpy as np

lc.set_license('my-license-key')

file_path = 'X_train.csv'
data = pd.read_csv(file_path)

columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
box_plot_data = [data[column].tolist() for column in columns]

dashboard = lc.Dashboard(
    rows=2,
    columns=3,
    theme=lc.Themes.Dark
)
def add_box_plot_to_chart(chart, column_data, column_name, column_index):
    q1 = np.percentile(column_data, 25)
    q3 = np.percentile(column_data, 75)
    median = np.median(column_data)
    min_val = np.min(column_data)
    max_val = np.max(column_data)
    
    series = chart.add_box_series()
    series.add(
        start=column_index - 0.4,
        end=column_index + 0.4,
        median=float(median),
        lower_quartile=float(q1),
        upper_quartile=float(q3),
        lower_extreme=float(min_val),
        upper_extreme=float(max_val)
    )
    series.set_name(column_name)

for i, (label, values) in enumerate(zip(columns, box_plot_data)):
    row_index = i // 3
    col_index = i % 3
    chart = dashboard.ChartXY(
        column_index=col_index,
        row_index=row_index,
        title=label
    )
    add_box_plot_to_chart(chart, values, label, i)
    x_axis = chart.get_default_x_axis()
    x_axis.set_title('Category')
    y_axis = chart.get_default_y_axis()
    y_axis.set_title('Values')

dashboard.open()

