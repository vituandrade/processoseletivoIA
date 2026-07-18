# Projeto 1 — Classificação MNIST

## 💻 O Desafio Técnico

Desenvolva um **modelo de Visão Computacional** capaz de **classificar dígitos manuscritos (0-9)**, e posteriormente **otimize-o para execução em dispositivos Edge**.

O foco não é apenas obter alta acurácia, mas **compreender o fluxo completo**:

**treinamento → validação → salvamento → conversão → otimização**

## 🎯 Conjunto de Dados

Dataset **MNIST**, disponível diretamente via `tf.keras.datasets.mnist` (não é necessário download manual).

## ✅ Requisitos Obrigatórios

### Etapa 1 — Treinamento do Modelo (`train_model.py`)

Implemente:

- Carregamento do dataset MNIST via TensorFlow
- **Split explícito treino/validação** (ex: `validation_split` ou um split manual)
- Construção de uma CNN com:
  - **3 a 4 blocos convolucionais** (`Conv2D` + `BatchNormalization` + `MaxPooling2D`)
  - Camada de `Dropout` antes da saída, para regularização
- Treinamento com **early stopping** baseado na perda de validação (`EarlyStopping`)
- Exibição da **acurácia de validação final** no terminal
- Salvamento do modelo treinado em formato Keras (`model.h5`)

### Etapa 2 — Otimização do Modelo (`optimize_model.py`)

Implemente:

- Carregamento do `model.h5` treinado
- Conversão para **TensorFlow Lite** (`model.tflite`)
- Aplicação de uma técnica de otimização (ex: **Dynamic Range Quantization**)

### Etapa 3 — Inferência com o Modelo Otimizado (`run_inference.py`)

Implemente:

- Carregamento especificamente do **`model.tflite`** (o artefato de edge — não
  o `model.h5`) usando `tf.lite.Interpreter`
- Execução de inferência em pelo menos **5 amostras** do conjunto de teste
- Exibição no terminal, para cada amostra, da classe **predita** vs. a classe **real**

> 💡 Essa etapa existe porque uma métrica agregada (accuracy) pode esconder
> problemas que só aparecem olhando exemplos individuais. Também é o teste mais
> próximo do uso real em produção: carregar o artefato de edge e classificar
> uma entrada por vez.

**Objetivo:** reduzir o tamanho do modelo, mantendo desempenho adequado para aplicações de Edge AI.

## 📂 Estrutura da Pasta

⚠️ Não altere os nomes dos arquivos.

```
projetos/1-classificacao-mnist/
├── train_model.py         # ✏️ Treinamento do modelo
├── optimize_model.py      # ✏️ Conversão e otimização
├── run_inference.py       # ✏️ Inferência de exemplo com o modelo otimizado
├── requirements.txt       # 📄 Dependências do projeto
├── model.h5               # 🤖 Gerado por você — deve ser commitado
├── model.tflite           # ⚡ Gerado por você — deve ser commitado
└── README.md               # 📝 Este arquivo (também usado como relatório)
```

## ⚠️ Restrições e Considerações de Engenharia

- Entrada do modelo: imagens 28x28, 1 canal (grayscale), normalizadas em [0, 1]
- CNN simples — evite arquiteturas muito profundas
- Não utilize modelos pré-treinados
- Número de épocas limitado (ex: até 15, com early stopping)
- Treinamento apenas em CPU

## ⚖️ Critérios de Avaliação

- **Funcionalidade** — execução correta dos scripts e geração dos arquivos `.h5` e `.tflite`
- **Qualidade do modelo** — acurácia de validação consistente com o esperado para o dataset
- **Edge AI** — conversão correta para `.tflite` com técnica de otimização aplicada
- **Documentação** — preenchimento adequado do relatório abaixo

---

## 📝 Relatório do Candidato

👤 **Nome Completo:** Victor Henrick Santos Andrade

### 1️⃣ Resumo da Arquitetura do Modelo

Implementei uma rede neural convolucional (CNN) sequencial composta por 3 blocos de extração de características. Cada bloco utiliza uma camada `Conv2D` (ativação ReLU), seguida de `BatchNormalization` (para estabilizar e acelerar o treinamento) e `MaxPooling2D` (para redução espacial). Antes da camada de saída densa com 10 neurônios (função softmax), foi aplicada uma camada de `Dropout` com taxa de 0.5 para atuar como regularizador e prevenir overfitting. O treinamento utilizou 10% dos dados para validação e empregou *Early Stopping* monitorando a `val_loss` com paciência de 3 épocas.

### 2️⃣ Bibliotecas Utilizadas

- TensorFlow (v2.21.0)
- NumPy (v2.5.1)
- Biblioteca padrão OS (Python)

### 3️⃣ Técnica de Otimização do Modelo

Foi aplicada a técnica de *Dynamic Range Quantization* nativa do TensorFlow Lite (`tf.lite.Optimize.DEFAULT`). Essa técnica converte os pesos do modelo de ponto flutuante de 32 bits para inteiros de 8 bits no momento da exportação, reduzindo drasticamente o uso de memória e armazenamento, otimizando o modelo para execução em dispositivos Edge sem perda significativa de precisão.

### 4️⃣ Resultados Obtidos

- **Acurácia de validação:** 0.9888 (98,88%)
- **Tamanho do model.h5:** 1.175 KB
- **Tamanho do model.tflite:** 104 KB

### 5️⃣ Comentários Adicionais (Opcional)

O pipeline funcionou perfeitamente, validando a eficácia do uso de Batch Normalization somado ao Dropout para redes aplicadas ao dataset MNIST. A conversão para TensorFlow Lite manteve a robustez do modelo na inferência.

### 6️⃣ Exemplo de Inferência

```text
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4