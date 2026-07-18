import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
# ---------------------------------------------------------------------------

def main():
    # 1 e 2. Carregar o dataset e normalizar
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    # Ajustar o shape para (28, 28, 1)
    x_train = tf.expand_dims(x_train, -1)
    x_test = tf.expand_dims(x_test, -1)

    # 4. Construir a CNN com 3 blocos Conv2D
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        
        # Bloco 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 3
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Camada de Dropout e Saída (10 classes)
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])

    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    # 5. Configurar EarlyStopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3, # Para o treinamento se a validação não melhorar por 3 épocas
        restore_best_weights=True
    )

    # 3. Treinar separando um conjunto de validação (10% dos dados)
    print("Iniciando treinamento da CNN...")
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=15,
        validation_split=0.1,
        callbacks=[early_stopping]
    )

    # 6. Exibir a acurácia de validação final no terminal
    # Pegamos a acurácia da última época registrada pelo early stopping
    val_acc = history.history['val_accuracy'][-1]
    print(f"\n[!] Treinamento concluído. Acurácia de validação final: {val_acc:.4f}")

    # 7. Salvar o modelo treinado
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    model.save(model_path)
    print(f"[!] Modelo salvo com sucesso em: {model_path}")

if __name__ == "__main__":
    main()