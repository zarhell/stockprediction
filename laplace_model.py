import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# Cargar los datos históricos
data = pd.read_csv('historical_prices.csv', encoding='ISO-8859-1')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Ordenar los datos y seleccionar precios de cierre
data = data.sort_index()
prices = data['Close']
dates = data.index
time = np.arange(len(prices))

# Calcular la transformada de Laplace
def laplace_transform(f, s):
    """
    Calcula la transformada de Laplace F(s) de una función discreta f(t).
    """
    return np.array([np.sum(f * np.exp(-s * t)) for t in range(len(f))])

# Interpolar los precios para generar una función continua
prices_interp = interp1d(time, prices, kind='cubic', fill_value="extrapolate")

# Transformada de Laplace para s en un rango definido
s_values = np.linspace(0.1, 1.0, 100)  # Rango de s
laplace_values = [quad(lambda t: prices_interp(t) * np.exp(-s * t), 0, len(prices))[0] for s in s_values]

# Guardar resultados en un archivo CSV
results = pd.DataFrame({'s': s_values, 'Laplace_Transform': laplace_values})
results.to_csv('laplace_transform.csv', index=False)

# Predicción futura basada en dinámica realista
future_days = 365
future_time = np.arange(len(prices), len(prices) + future_days)

# Término no homogéneo f(t) para simular perturbaciones externas
def external_forces(t):
    """
    Simula fuerzas externas con ruido aleatorio y una señal sinusoidal.
    """
    return 0.01 * np.sin(0.05 * t) + np.random.normal(0, 0.2, len(t))

# Predicción con dinámica oscilatoria
def predict_prices_dynamic(t, P0, V0, omega, damping):
    """
    Predicción basada en una solución oscilatoria amortiguada con fuerzas externas.
    """
    C1 = P0
    C2 = (V0 + (damping / 2) * P0) / omega if omega > 0 else 0
    if omega > 0:
        return (C1 * np.exp(-damping / 2 * t) * np.cos(omega * t) +
                C2 * np.exp(-damping / 2 * t) * np.sin(omega * t) +
                external_forces(t))
    else:
        # Solución para un sistema sobreamortiguado
        return C1 * np.exp(-damping / 2 * t) + external_forces(t)

# Parámetros para la ecuación diferencial
P0 = prices.values[-1]  # Último precio conocido
V0 = np.gradient(prices.values)[-1]  # Última velocidad conocida
damping = 0.1  # Factor de amortiguamiento
omega = 0.5  # Frecuencia angular ajustada

# Generar predicciones futuras
predicted_prices_dynamic = predict_prices_dynamic(future_time - future_time[0], P0, V0, omega, damping)

# Guardar datos predichos
future_dates = pd.date_range(start=dates[-1] + pd.Timedelta(days=1), periods=future_days)
predictions = pd.DataFrame({'Date': future_dates, 'Predicted_Price': predicted_prices_dynamic})
predictions.to_csv('predicted_prices_dynamic.csv', index=False)

# Visualización de resultados
plt.figure(figsize=(12, 6))
plt.plot(dates, prices, label='Datos históricos', color='blue')
plt.plot(future_dates, predicted_prices_dynamic, label='Predicciones dinámicas', linestyle='--', color='red')
plt.xlabel("Fecha")
plt.ylabel("Precio")
plt.legend()
plt.title("Transformada de Laplace y Predicción de Precios con Dinámica Realista")
plt.grid()
plt.show()

print("Archivos generados: 'laplace_transform.csv' y 'predicted_prices_dynamic.csv'")
