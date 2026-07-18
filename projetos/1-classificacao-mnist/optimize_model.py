import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    tflite_path = os.path.join(script_dir, "model.tflite")

    # 1. Carregar o modelo treinado
    print("Carregando o modelo original (model.h5)...")
    model = tf.keras.models.load_model(model_path)

    # 2. Iniciar o conversor do TensorFlow Lite
    print("Preparando a conversão para TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # 3. Aplicar a técnica de otimização (Dynamic Range Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # 4. Executar a conversão e salvar como "model.tflite"
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    print(f"[!] Modelo otimizado para Edge AI salvo em: {tflite_path}")

if __name__ == "__main__":
    main()