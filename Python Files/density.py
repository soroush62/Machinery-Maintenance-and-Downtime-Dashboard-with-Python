import lightningchart as lc
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

lc.set_license('my-license-key')

file_path = 'X_train.csv'
data = pd.read_csv(file_path)


columns = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

density_data = {col: gaussian_kde(data[col].dropna()) for col in columns}
x_vals = {col: np.linspace(data[col].min(), data[col].max(), 100) for col in columns}
y_vals = {col: density_data[col](x_vals[col]) for col in columns}

dashboard = lc.Dashboard(
    rows=2,
    columns=3,
    theme=lc.Themes.Black  
)
def add_density_plot_to_chart(chart, x, y, label, color):
    series = chart.add_positive_area_series()
    series.add(x.tolist(), y.tolist())
    series.set_name(label)
    series.set_fill_color(color)
    x_axis = chart.get_default_x_axis()
    y_axis = chart.get_default_y_axis()
    x_axis.set_title('Values')
    y_axis.set_title('Density')
    chart.add_legend()


colors = [
    lc.Color('#FF5733'),  # Red
    lc.Color('#33FF57'),  # Green
    lc.Color('#3357FF'),  # Blue
    lc.Color('#FF33A8'),  # Pink
    lc.Color('#33FFF5')   # Cyan
]

for i, column in enumerate(columns):
    row_index = i // 3
    col_index = i % 3
    chart = dashboard.ChartXY(
        column_index=col_index,
        row_index=row_index,
        title=column
    )
    add_density_plot_to_chart(chart, x_vals[column], y_vals[column], column, colors[i])

dashboard.open()

