# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 10:38:26 2025

@author: rportatil113

"""

%reset -f  

import os
import numpy as np  # Para cálculos numéricos
import pandas as pd
import statsmodels.api as sm  # Para el modelo lineal con estadísticas
from sklearn.linear_model import LinearRegression  # Modelo lineal con sklearn
from sklearn.metrics import mean_squared_error, r2_score  # Evaluación del modelo
import matplotlib.pyplot as plt  



# Cambiar el directorio de trabajo
os.chdir(r"C:\Users\rportatil113\Documents\analisis Repaso")

# Cargar los datos
df = pd.read_csv("Auto.txt", delimiter=",")  

# Generar los datos
X = df["mpg"]  # Variable predictora
Y = df["horsepower"].astype(float)  # Variable dependiente

# Convertir a DataFrame
data = pd.DataFrame({'X': X, 'Y': Y})

# --- Modelo con statsmodels ---
X_sm = sm.add_constant(X)  # Agregar intercepto
model_sm = sm.OLS(Y, X_sm).fit()

# Resultados
print("Resultados con statsmodels:")
print(model_sm.summary())

# Otras opciones

# Predicción para X = 11 con intervalos de confianza
valor = np.array([[1, 11]])  # Agregar constante
pred_11 = model_sm.get_prediction(valor)
pred_summary = pred_11.summary_frame(alpha=0.05)

print(model_sm.summary())
print(f"Coeficientes del modelo (statsmodels): Intercepto: {model_sm.params[0]:.4f}, Pendiente: {model_sm.params[1]:.4f}")
print(f"Predicción para X = 1: {pred_summary['mean'][0]:.4f}")
print(f"Intervalo de confianza del 95%: ({pred_summary['mean_ci_lower'][0]:.4f}, {pred_summary['mean_ci_upper'][0]:.4f})")
print(f"Intervalo de predicción del 95%: ({pred_summary['obs_ci_lower'][0]:.4f}, {pred_summary['obs_ci_upper'][0]:.4f})")

# --- Modelo con sklearn ---
X_sklearn = X.values.reshape(-1, 1)  # Ajustar dimensiones
model_sklearn = LinearRegression().fit(X_sklearn, Y)

# Predicciones
Y_pred = model_sklearn.predict(X_sklearn)

# Cálculo de RSE (Residual Standard Error)
residuals = Y - Y_pred

RSS = np.sqrt(np.sum(residuals**2))
              
RSE = np.sqrt(RSS/(len(Y)-2))  

print("\nResultados con sklearn:")
print(f"Intercepto (beta0): {model_sklearn.intercept_:.4f}")
print(f"Pendiente (beta1): {model_sklearn.coef_[0]:.4f}")
print(f"R^2: {r2_score(Y, Y_pred):.4f}")
print(f"RSE: {RSE:.4f}")

# --- Gráfica de la regresión ---
plt.scatter(X, Y, label="Datos")
plt.plot(X, Y_pred, color="red", label="Regresión lineal")
plt.xlabel("MPG")
plt.ylabel("Horsepower")
plt.legend()
plt.title("Regresión lineal: Horsepower vs MPG")
plt.show()

# --- Gráficos de diagnóstico ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].scatter(Y_pred, residuals)
axs[0, 0].axhline(y=0, color='r', linestyle='--')
axs[0, 0].set_xlabel("Valores ajustados")
axs[0, 0].set_ylabel("Residuos")
axs[0, 0].set_title("Residuos vs Valores ajustados")

axs[0, 1].hist(residuals, bins=20, edgecolor='black')
axs[0, 1].set_title("Histograma de residuos")

sm.qqplot(residuals, line="45", fit=True, ax=axs[1, 0])
axs[1, 0].set_title("Gráfico Q-Q de residuos")

axs[1, 1].scatter(Y_pred, np.abs(residuals))
axs[1, 1].set_xlabel("Valores ajustados")
axs[1, 1].set_ylabel("|Residuos|")
axs[1, 1].set_title("Escala de localización")

plt.tight_layout()
plt.show()

"""
preguntas
1-observa los coefiicientes del modelo y saca conclusiones
es negativa, significa que a medida que aumenta el MPG (consumo de combustible), el Horsepower (potencia) disminuye.
 Esto tiene sentido porque los autos con mayor potencia suelen consumir más combustible y tener menos MPG.
2.¿existe realcion entre ambas variables?
Sí, claramente hay una relación entre MPG y Horsepower. El gráfico muestra que a medida que aumenta el MPG,
 el Horsepower disminuye, lo que indica una correlación negativa.
3. como de significativa es la realcion

4. la realciones es positiva o negativa
negariva, podemos observar como la linea de regresion tiene pendiente descendente lo que quiere decir que
a medida que el MPG aumenta (es decir, el auto es más eficiente en consumo), el Horsepower disminuye.
5. cual es el valor de la prediccion para un valor de horsepowe=98


"""
horsepower_value = 98
mpg_predicho = model_sklearn.predict([[horsepower_value]])
print(f"Predicción de MPG para horsepower=98: {mpg_predicho[0]:.2f}")
print(model_sm.summary())


X = df[["weight", "horsepower", "displacement"]]
Y = df["mpg"]  # Variable dependiente

# Revisar si hay valores nulos y eliminarlos si existen
df = df.dropna(subset=["mpg", "weight", "horsepower", "displacement"])
X = df[["weight", "horsepower", "displacement"]].astype(float)
Y = df["mpg"].astype(float)

# --- Modelo con statsmodels ---
X_sm = sm.add_constant(X)  # Agregar intercepto
model_sm = sm.OLS(Y, X_sm).fit()

# Resultados del modelo
print(model_sm.summary())

# --- Modelo con sklearn ---
model_sklearn = LinearRegression().fit(X, Y)

# Predicciones
Y_pred = model_sklearn.predict(X)

# Métricas del modelo
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("\nResultados con sklearn:")
print(f"Intercepto: {model_sklearn.intercept_:.4f}")
print(f"Coeficientes: {model_sklearn.coef_}")
print(f"R^2: {r2:.4f}")
print(f"Error cuadrático medio (MSE): {mse:.4f}")

# --- Gráfica de predicciones ---
plt.figure(figsize=(8, 5))
plt.scatter(Y, Y_pred, alpha=0.7)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color="red", linestyle="--")
plt.xlabel("MPG Real")
plt.ylabel("MPG Predicho")
plt.title("MPG Real vs. MPG Predicho")
plt.show()