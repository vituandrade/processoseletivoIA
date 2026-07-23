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
