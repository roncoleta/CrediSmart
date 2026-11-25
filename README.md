# CrediSmart: Sistema Híbrido de Decisão de Crédito

## Integrantes

Leonardo Nogueira RA: 2401360
Victor Roncoleta RA: 2410270

## O que o projeto faz

Pipeline de decisão de crédito em três etapas:

1. **MLP** estima a probabilidade de inadimplência.
2. **Controlador fuzzy Mamdani** usa essa probabilidade e o comprometimento de renda para recomendar o limite de crédito (% do limite máximo).
3. **Algoritmo genético** ajusta neurônios ocultos, taxa de aprendizado e épocas da MLP.

## Arquivos

`CrediSmart.ipynb`: Notebook com as 4 partes em sequência (texto explicativo + código)
`CrediSmart.py` Mesmo código em um script único
`Relatorio CrediSmart.pdf` Relatório técnico
`fig_*.png` Gráficos gerados pelo código

## Como executar

Requisitos: Python 3.10 ou mais novo, com `numpy`, `matplotlib` e `scikit-learn`.

pip install numpy matplotlib scikit-learn

**Opção 1, notebook:** abrir `CrediSmart.ipynb` e executar todas as células, de cima para baixo (Run All).

**Opção 2, script:**

python CrediSmart.py

O código roda do início ao fim sem erro e leva pouco mais de 1 minuto (a maior parte é o algoritmo genético, que treina cerca de 250 redes). Os gráficos são salvos como arquivos `fig_*.png` na pasta onde o código é executado. No notebook eles também aparecem abaixo de cada célula.