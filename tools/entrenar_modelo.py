import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# ---------- Rutas ----------
CARPETA = os.path.dirname(os.path.abspath(__file__))
RAIZ = CARPETA   # dataset.csv e include/ están en la misma carpeta que este script
CSV = os.path.join(RAIZ, "dataset.csv")
INCLUDE = os.path.join(RAIZ, "include")

# Umbral de confianza: MISMO valor que usa inferir.cpp (0.70f).
# Si la probabilidad más alta es menor a esto, el resultado es "incierto".
UMBRAL_CONFIANZA = 0.70

# ---------- 1. Cargar los datos ----------
df = pd.read_csv(CSV, header=None)

# La primera columna es la etiqueta; las otras 384 son los datos
clases = sorted(df[0].unique())
print("Clases:", clases)
print(df[0].value_counts())

# Convertir etiquetas de texto a números (lumos -> 0, etc.)
y = np.array([clases.index(nombre) for nombre in df[0]])

# Convertir los datos a forma (muestras, 64 pasos, 6 ejes)
X = df.iloc[:, 1:].values.astype("float32").reshape(-1, 64, 6)

# Normalizar: MISMA escala que usa el ESP32 al inferir
X[:, :, :3] /= 4.0     # acelerometro (±4 g)
X[:, :, 3:] /= 500.0   # giroscopio (±500 °/s)

# ---------- 2. Separar entrenamiento y prueba ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

# ---------- 3. Aumentar datos (crear variaciones) ----------
def aumentar(X, y, copias=8):
    lista_X = [X]
    lista_y = [y]
    for _ in range(copias):
        nuevo = X * np.random.uniform(0.85, 1.15, (len(X), 1, 1))  # cambia la fuerza
        nuevo = nuevo + np.random.normal(0, 0.02, X.shape)         # añade ruido
        lista_X.append(nuevo.astype("float32"))
        lista_y.append(y)
    return np.concatenate(lista_X), np.concatenate(lista_y)

X_train, y_train = aumentar(X_train, y_train)

# ---------- 4. Crear el modelo ----------
modelo = tf.keras.Sequential([
    layers.Input(shape=(64, 6)),
    layers.Conv1D(16, 5, padding="same", activation="relu"),
    layers.MaxPool1D(2),
    layers.Conv1D(32, 5, padding="same", activation="relu"),
    layers.MaxPool1D(2),
    layers.Conv1D(32, 3, padding="same", activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.3),
    layers.Dense(len(clases), activation="softmax"),
])

modelo.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"],)

# ---------- 5. Entrenar (con umbral de confianza del 70%) ----------
def evaluar_con_umbral(modelo, X, y, umbral=UMBRAL_CONFIANZA):
    """Evalúa igual que el ESP32: si la confianza < umbral, es 'incierto'."""
    probs = modelo.predict(X, verbose=0)
    prediccion = probs.argmax(axis=1)
    confianza = probs.max(axis=1)

    seguras = confianza >= umbral                       # pasan el 70%
    cobertura = seguras.mean()                          # % de gestos que el ESP32 aceptaria
    if seguras.any():
        precision_segura = (prediccion[seguras] == y[seguras]).mean()
    else:
        precision_segura = 0.0
    return precision_segura, cobertura, probs


class ReporteConfianza(tf.keras.callbacks.Callback):
    """Cada 5 épocas muestra cómo iría el modelo en el ESP32 con el umbral del 70%."""
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % 5 == 0:
            p, c, _ = evaluar_con_umbral(self.model, X_test, y_test)
            print(f"   [conf>={UMBRAL_CONFIANZA:.0%}] precisión segura: {p * 100:.1f}% | "
                  f"gestos aceptados: {c * 100:.1f}%")


modelo.fit(X_train, y_train, epochs=60, batch_size=32,
           validation_data=(X_test, y_test), callbacks=[ReporteConfianza()])

perdida, precision = modelo.evaluate(X_test, y_test, verbose=0)
print(f"\nPrecisión normal en datos de prueba: {precision * 100:.1f}%")

precision_segura, cobertura, probs = evaluar_con_umbral(modelo, X_test, y_test)
print(f"Con umbral de confianza {UMBRAL_CONFIANZA:.0%}:")
print(f"  - Gestos aceptados (confianza >= {UMBRAL_CONFIANZA:.0%}): {cobertura * 100:.1f}%")
print(f"  - Gestos 'inciertos' (descartados):  {(1 - cobertura) * 100:.1f}%")
print(f"  - Precisión de los gestos aceptados:  {precision_segura * 100:.1f}%")

# ---------- 6. Convertir a TFLite ----------
convertidor = tf.lite.TFLiteConverter.from_keras_model(modelo)
datos_tflite = convertidor.convert()
print("Tamaño del modelo:", len(datos_tflite), "bytes")

# ---------- 7. Generar include/model.h ----------
os.makedirs(INCLUDE, exist_ok=True)
with open(os.path.join(INCLUDE, "model.h"), "w") as f:
    f.write("#pragma once\n")
    f.write("alignas(8) const unsigned char model_tflite[] = {\n")
    for i, byte in enumerate(datos_tflite):
        f.write(f"0x{byte:02x},")
        if (i + 1) % 16 == 0:
            f.write("\n")
    f.write("\n};\n")

# ---------- 8. Generar include/clases.h ----------
with open(os.path.join(INCLUDE, "clases.h"), "w") as f:
    f.write("#pragma once\n")
    nombres = ", ".join(f'"{c}"' for c in clases)
    f.write(f"const char* CLASES[] = {{{nombres}}};\n")
    f.write(f"const int N_CLASES = {len(clases)};\n")

print("\nListo. Se crearon include/model.h e include/clases.h")