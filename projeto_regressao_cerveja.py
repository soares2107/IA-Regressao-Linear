# Projeto de Regressão Linear - Previsão de Consumo de Cerveja
# ================================================================
# Autor: [Joao Gabriel Soares Pereira da Silva]
#Disciplina : Inteligencia Artificial
# Dataset: beer_consuption.csv

# === (a) Importar bibliotecas ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# === (b) Leitura do arquivo ===
cerveja = pd.read_csv("beer_consuption.csv")

# === (c) Primeiras observações ===
pd.set_option('display.max_columns', None)
print("(c) Primeiras 5 linhas:")
print(cerveja.head())

# === (d) Últimas observações ===
pd.set_option('display.max_columns', None)
print("\n(d) Últimas 5 linhas:")
print(cerveja.tail())

# === (e) Dimensão do dataset ===
print("\n(e) Dimensão da base:")
print(cerveja.shape)

# === (f) Verificação de valores faltantes ===
print("\n(f) Valores faltantes:")
print(cerveja.isna().sum())

# === (g) Tipos das variáveis ===
print("\n(g) Tipos das variáveis:")
print(cerveja.dtypes)

# === (h) Correlação entre variáveis (Pearson) ===
print("\n(h) Matriz de Correlação:")
print(cerveja.select_dtypes(include=[np.number]).corr())


# === (i) Estatística descritiva ===
print("\n(i) Tabela descritiva:")
print(cerveja.describe())

# === (j) Gráfico de barras para Final de Semana ===
sns.countplot(x="Final de Semana", data=cerveja, palette=["blue", "green"])
plt.title("Letra J")
plt.show()


# === (k) Gráfico de linhas - Temperaturas e Consumo ===
plt.plot(cerveja["Temperatura Media (C)"], label="Temperatura Média (C)", color="blue")
plt.plot(cerveja["Temperatura Minima (C)"], label="Temperatura Mínima (C)", color="green")
plt.plot(cerveja["Temperatura Maxima (C)"], label="Temperatura Máxima (C)", color="red")
plt.plot(cerveja["Consumo de cerveja (litros)"], color="black")
plt.title("Letra K - Temperaturas e Consumo")
plt.legend()
plt.show()

# === (l) Gráfico de Precipitação ===
plt.plot(cerveja["Precipitacao (mm)"])
plt.title("Letra L - Precipitação Diária")
plt.show()

# === (m) Gráfico de Consumo ===
plt.plot(cerveja["Consumo de cerveja (litros)"], color ='black')
plt.title("Letra M - Consumo de Cerveja")
plt.show()

# === (n) Correlograma Pearson ===
plt.figure(figsize=(10, 8))  # Ajuste o tamanho conforme necessário

sns.heatmap(cerveja.corr(numeric_only=True), annot=True,
            fmt=".2f", cmap="RdYlGn", vmin=-0.2, vmax=1)

plt.title("Letra N - Correlação de Pearson")
plt.tight_layout()  # Garante que nada fique cortado
plt.show()


# === (o) Correlograma Spearman ===
plt.figure(figsize=(10, 8))
sns.heatmap(cerveja.corr(method='spearman', numeric_only=True),
            annot=True, fmt=".2f", cmap="RdYlGn", vmin=-0.2, vmax=1)
plt.title("Letra O - Correlação de Spearman")
plt.tight_layout()
plt.show()

# === (p) Boxplots ===
plt.figure(figsize=(15, 6))
sns.boxplot(
    data=cerveja[[
        "Temperatura Media (C)",
        "Temperatura Minima (C)",
        "Temperatura Maxima (C)",
        "Precipitacao (mm)",
        "Consumo de cerveja (litros)"
    ]],
    showfliers=False,
    boxprops=dict(facecolor='none', edgecolor="blue"),
    medianprops=dict(color="red")
)
plt.title("Letra P - Boxplots")
plt.grid(True)
plt.show()



# === (q) Histogramas ===
colunas = ["Temperatura Media (C)", "Temperatura Minima (C)", "Temperatura Maxima (C)",
           "Precipitacao (mm)", "Consumo de cerveja (litros)"]
plt.figure(figsize=(12, 8))
for i, col in enumerate(colunas):
    plt.subplot(3, 2, i+1)
    plt.hist(cerveja[col], bins=30)
    plt.title(col)
plt.tight_layout()
plt.show()

# === (r) Gráfico de dispersão: Consumo x variáveis ===
variaveis = ["Temperatura Media (C)", "Temperatura Minima (C)", "Temperatura Maxima (C)", "Precipitacao (mm)"]
plt.figure(figsize=(12, 8))
for i, var in enumerate(variaveis):
    plt.subplot(2, 2, i+1)
    plt.scatter(cerveja["Consumo de cerveja (litros)"], cerveja[var])
    plt.xlabel("Consumo de cerveja (litros)")
    plt.ylabel(var)
plt.tight_layout()
plt.show()


# === PREPARAR DADOS PARA REGRESSÃO ===
X = cerveja["Temperatura Media (C)"].values
y = cerveja["Consumo de cerveja (litros)"].values

# === (s-a) Regressão Linear - Mínimos Quadrados ===
mean_x = X.mean()
mean_y = y.mean()
b1_mmq = ((X - mean_x) * (y - mean_y)).sum() / ((X - mean_x) ** 2).sum()
b0_mmq = mean_y - b1_mmq * mean_x
y_pred_mmq = b0_mmq + b1_mmq * X

# === (s-b) Regressão Linear - Gradiente Descendente ===
b0_gd = 0
b1_gd = 0
alpha = 0.001
epochs = 10000
for _ in range(epochs):
    y_pred = b0_gd + b1_gd * X
    erro = y - y_pred
    b0_gd += alpha * erro.mean()
    b1_gd += alpha * (erro * X).mean()
y_pred_gd = b0_gd + b1_gd * X

# === (s-c) Comparar visualmente ===
plt.scatter(X, y, label="Dados Reais", alpha=0.6)
plt.plot(X, y_pred_mmq, color="blue", label="MMQ")
plt.plot(X, y_pred_gd, color="red", linestyle="--", label="Gradiente Descendente")
plt.xlabel("Temperatura Média (C)")
plt.ylabel("Consumo de cerveja (litros)")
plt.title("Letra S - Comparativo de Modelos")
plt.legend()
plt.grid(True)
plt.show()

# === (t) Métricas de Avaliação ===
def calcular_metricas(y, y_pred):
    n = len(y)
    p = 1
    ss_total = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    r2 = 1 - ss_res / ss_total
    r2_ajustado = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    mse = np.mean((y - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100
    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y)) ** 2))
    return r2, r2_ajustado, mse, rmse, mae, mape, rmsle

metricas_mmq = calcular_metricas(y, y_pred_mmq)
metricas_gd = calcular_metricas(y, y_pred_gd)

print("\n(t) Métricas - MMQ:")
print("R²: {:.4f}\nR² ajustado: {:.4f}\nMSE: {:.4f}\nRMSE: {:.4f}\nMAE: {:.4f}\nMAPE: {:.2f}%\nRMSLE: {:.4f}".format(*metricas_mmq))

print("\n(t) Métricas - Gradiente Descendente:")
print("R²: {:.4f}\nR² ajustado: {:.4f}\nMSE: {:.4f}\nRMSE: {:.4f}\nMAE: {:.4f}\nMAPE: {:.2f}%\nRMSLE: {:.4f}".format(*metricas_gd))
