import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Cargar los datos
data = pd.read_csv('historical_prices.csv', encoding='ISO-8859-1')
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Ordenar Datos
data = data.sort_index()
#Usar el precio de cierre
prices = data['Close']  

# Cálculo de la derivada primera y segunda
prices_np = prices.values
time = np.arange(len(prices_np))

# Derivada primera (velocidad)
velocity = np.gradient(prices_np, time)

# Derivada segunda (aceleración)
acceleration = np.gradient(velocity, time)

# la ecuación diferencial de segundo orden
def model_function(t, a, b):
    return acceleration + a * velocity + b * prices_np

# Ajuste de curva para determinar los valores de 'a' y 'b'
popt, pcov = curve_fit(lambda t, a, b: model_function(t, a, b), time, prices_np, p0=[1, 1])

a, b = popt
print(f"Valor de a: {a}")
print(f"Valor de b: {b}")

# Paso 5: Visualización de los resultados
plt.figure(figsize=(12, 6))
plt.plot(time, prices_np, label='Precio real')
plt.plot(time, model_function(time, a, b), label='Modelo ajustado', linestyle='--')
plt.xlabel("Tiempo")
plt.ylabel("Precio")
plt.legend()
plt.title("Predicción de precios de acciones utilizando ecuación diferencial de segundo orden")
plt.show()
