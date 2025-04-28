# 📈 Projeto de Regressão Linear - Previsão de Consumo de Cerveja

Este projeto aplica técnicas de **Regressão Linear** para analisar a relação entre variáveis climáticas (temperatura e precipitação) e o **consumo de cerveja**.

> **Autor:** João Gabriel Soares Pereira da Silva  
> **Disciplina:** Inteligência Artificial

---

## 📚 Sobre o Projeto

O objetivo principal é construir e comparar dois modelos de regressão:
- **Mínimos Quadrados Ordinários (MMQ)**
- **Gradiente Descendente**

Utilizando como variável preditora a **Temperatura Média** e como variável resposta o **Consumo de cerveja (litros)**.

Além da modelagem, o projeto inclui:
- Análise exploratória de dados (EDA)
- Gráficos de dispersão, histogramas, boxplots
- Correlogramas de Pearson e Spearman
- Avaliação dos modelos com métricas como R², RMSE, MAE, MAPE e RMSLE.

---

## 📁 Estrutura do Projeto

- `beer_consuption.csv` — Base de dados contendo informações climáticas e consumo de cerveja.
- `projeto_regressao_cerveja.py` — Script principal que realiza a análise exploratória, modelagem e avaliação dos resultados.

---

## 🛠️ Tecnologias Utilizadas

- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 🚀 Como Executar o Projeto

1. Clone o repositório:

```bash
git clone https://github.com/soares2107/IA-Regressao-Linear.git
cd IA-Regressao-Linear
```

2. Instale as bibliotecas necessárias:

```bash
pip install pandas numpy matplotlib seaborn
```

3. Execute o script:

```bash
python projeto_regressao_cerveja.py
```

---

## 📊 Principais Análises Realizadas

- **Distribuição dos dados:** histogramas, boxplots
- **Análise de correlação:** correlogramas de Pearson e Spearman
- **Gráficos de linha:** comportamento das temperaturas e do consumo
- **Modelos de regressão:** comparação visual e numérica entre MMQ e Gradiente Descendente
- **Métricas de avaliação:** R², RMSE, MAE, MAPE, RMSLE

---

## 🧐 Resultados Obtidos

Ambos os métodos apresentaram boa capacidade de ajuste, com pequenas diferenças entre as métricas. O comparativo visual e quantitativo dos modelos é apresentado no final da execução do script.

---

## 📄 Licença

Este projeto é apenas para fins educacionais.
