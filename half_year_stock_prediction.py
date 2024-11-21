import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar los datos
data = pd.read_csv('historical_prices.csv', encoding='ISO-8859-1')
data['Date'] = pd.to_datetime(data['Date'], format='%d-%b-%y', errors='coerce')
data.set_index('Date', inplace=True)

# Ordenar datos
data = data.sort_index()
prices = data['Close']
dates = data.index
time = np.arange(len(prices)).reshape(-1, 1)  # Tiempo como variable independiente para la regresión

# Ajuste de regresión lineal usando los datos históricos
model = LinearRegression()
model.fit(time, prices)  # Entrenar el modelo con tiempo y precios históricos

# Generar predicciones para el siguiente semestre (180 días)
future_days = 180
future_time = np.arange(len(prices), len(prices) + future_days).reshape(-1, 1)
predicted_prices = model.predict(future_time)

# Generar fechas futuras
future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_days)

# Crear DataFrame para predicciones
predictions_df = pd.DataFrame({
    'Date': future_dates,
    'Price': predicted_prices,
    'Type': 'Predicción'
})

# Crear DataFrame para datos históricos
historical_data = data[['Close']].reset_index()
historical_data.rename(columns={'Close': 'Price'}, inplace=True)
historical_data['Type'] = 'Histórico'

# Combinar datos históricos y predicciones
combined_data = pd.concat([historical_data, predictions_df], ignore_index=True) 

# Guardar el DataFrame combinado en un archivo CSV
combined_data.to_csv('combined_prices.csv', index=False)
print("Datos combinados guardados en 'combined_prices.csv'")

# Visualización de resultados
plt.figure(figsize=(12, 6))
for data_type, group_data in combined_data.groupby('Type'):
    plt.plot(group_data['Date'], group_data['Price'], label=data_type)

plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.title("Visualización de Datos Históricos y Predicciones")
plt.grid()
plt.show()
