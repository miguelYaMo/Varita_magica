import os
import csv
import serial

ETIQUETAS = ["Confundo","Sectumsempra","Accio","Lumos","Nada"]
PUERTO = "/dev/ttyUSB0"   # En Windows seria "COM3"

CARPETA = os.path.dirname(os.path.abspath(__file__))
ARCHIVO = os.path.join(CARPETA, "..", "dataset.csv")

VALORES_ESPERADOS = 64 * 6   # 64 pasos x 6 ejes = 384 números

# Contar cuantas muestras hay ya de cada etiqueta
conteo = {e: 0 for e in ETIQUETAS}
if os.path.exists(ARCHIVO):
    with open(ARCHIVO, newline="") as f:
        for fila in csv.reader(f):
            if fila and fila[0] in conteo:
                conteo[fila[0]] += 1

ser = serial.Serial(PUERTO, 115200, timeout=1)
ser.reset_input_buffer()
print("Listo. Mantén el botón, haz el hechizo y suéltalo.")

archivo = open(ARCHIVO, "a", newline="")
escritor = csv.writer(archivo)

try:
    while True:
        # Leer una linea del ESP32
        linea = ser.readline().decode(errors="ignore").strip()

        # Solo nos interesan las lineas que empiezan con "G,"
        if not linea.startswith("G,"):
            continue

        valores = linea.split(",")[1:]   
        if len(valores) != VALORES_ESPERADOS:
            print("Gesto incompleto, lo ignoro")
            continue

        # Menu
        print("\n>>> Gesto recibido. ¿Qué hechizo era?")
        for i, nombre in enumerate(ETIQUETAS):
            print(f"  [{i}] {nombre}  ({conteo[nombre]} muestras)")
        print("  [Enter] descartar    [q] salir")

        respuesta = input("Tu elección: ").strip().lower()

        if respuesta == "q":
            break
        elif respuesta.isdigit() and int(respuesta) < len(ETIQUETAS):
            etiqueta = ETIQUETAS[int(respuesta)]
            escritor.writerow([etiqueta] + valores)
            archivo.flush()
            conteo[etiqueta] += 1
            print(f"Guardado como '{etiqueta}'")
        else:
            print("Descartado")

        ser.reset_input_buffer()

except KeyboardInterrupt:
    pass

archivo.close()
ser.close()
print("\nResumen:", conteo)