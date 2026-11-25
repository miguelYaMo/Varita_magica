#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "gesture_core.h"   // MPU, boton y remuestreo
#include "model.h"          // el modelo (lo genera entrenar.py)
#include "clases.h"         // los nombres de hechizos (lo genera entrenar.py)

// EDITA AQUI TU WIFI (debe ser de 2.4 GHz) 
const char* WIFI_NOMBRE = "MAGIA";
const char* WIFI_CLAVE  = "ABCD1234";

// Memoria de trabajo del modelo. Si falla, subela a 80 * 1024.
const int TAMANO_ARENA = 60 * 1024;
uint8_t arena[TAMANO_ARENA];

tflite::MicroMutableOpResolver<10> resolver;
tflite::MicroInterpreter* interprete;
TfLiteTensor* entrada;
TfLiteTensor* salida;

float probabilidades[20];   // resultado de la ultima prediccion
WebServer servidor(80);

// Pagina web (el navegador la pide cada 0.4 s al ESP32)
const char PAGINA[] PROGMEM = R"HTML(
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Varita mágica</title>
</head>
<body style="font-family:sans-serif; background:#111; color:#eee; max-width:500px; margin:auto; padding:10px">
  <h2>🪄 Hechizo detectado</h2>
  <div id="resultado">Haz un gesto...</div>
  <script>
    async function actualizar() {
      const respuesta = await fetch('/predict');
      const datos = await respuesta.json();
      let html = '';
      for (let i = 0; i < datos.hechizos.length; i++) {
        const p = datos.probs[i] * 100;
        html += '<p>' + datos.hechizos[i] + ': ' + p.toFixed(1) + '%' +
                '<div style="background:#6c5ce7; height:14px; width:' + p + '%"></div></p>';
      }
      document.getElementById('resultado').innerHTML = html;
    }
    setInterval(actualizar, 400);
    actualizar();
  </script>
</body>
</html>
)HTML";

// Responde a /predict con las probabilidades en formato JSON
void enviarPrediccion() {
  String json = "{\"hechizos\":[";
  for (int i = 0; i < N_CLASES; i++) {
    json += "\"" + String(CLASES[i]) + "\"";
    if (i < N_CLASES - 1) json += ",";
  }
  json += "],\"probs\":[";
  for (int i = 0; i < N_CLASES; i++) {
    json += String(probabilidades[i], 4);
    if (i < N_CLASES - 1) json += ",";
  }
  json += "]}";
  servidor.send(200, "application/json", json);
}
// Prepara el modelo
void iniciarModelo() {
  const tflite::Model* modelo = tflite::GetModel(model_tflite);

  // Operaciones que usa nuestro modelo
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddReshape();
  resolver.AddExpandDims();
  resolver.AddSqueeze();

  //static tflite::MicroInterpreter interprete_estatico(modelo, resolver, arena, TAMANO_ARENA);
  static tflite::MicroErrorReporter reporter_errores;
  static tflite::MicroInterpreter interprete_estatico(modelo, resolver, arena, TAMANO_ARENA, &reporter_errores);
  
  interprete = &interprete_estatico;

  if (interprete->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors fallo (falta memoria o falta una operacion)");
    while (true) delay(1000);   // se queda parado
  }

  entrada = interprete->input(0);
  salida = interprete->output(0);
  Serial.println("Modelo cargado correctamente");
}
// Recibe un gesto, lo pasa al modelo y guarda las probabilidades
void predecir(float gesto[N_OUT][6]) {
  // Copiar el gesto a la entrada del modelo, con la MISMA normalizacion del entrenamiento
  for (int i = 0; i < N_OUT; i++) {
    for (int k = 0; k < 6; k++) {
      float escala = (k < 3) ? 4.0f : 500.0f;
      entrada->data.f[i * 6 + k] = gesto[i][k] / escala;
    }
  }

  interprete->Invoke();

  // Leer el resultado y mostrarlo en el monitor serie
  Serial.println("--- Prediccion ---");
  for (int i = 0; i < N_CLASES; i++) {
    probabilidades[i] = salida->data.f[i];
    Serial.print(CLASES[i]);
    Serial.print(": ");
    Serial.println(probabilidades[i], 3);
  }

  int mejor = 0;
  for (int i = 1; i < N_CLASES; i++)
    if (probabilidades[i] > probabilidades[mejor]) mejor = i;

  if (probabilidades[mejor] < 0.70f) {
    Serial.println(">> Resultado: incierto");
  } else if (strcmp(CLASES[mejor], "Nada") == 0) {
    Serial.println(">> Resultado: nada");
  } else {
    Serial.print(">> Hechizo: ");
    Serial.println(CLASES[mejor]);
  }

}

void setup() {
  Serial.begin(115200);
  gestureInit();
  iniciarModelo();

  // Conectar al WiFi
  /*
  WiFi.begin(WIFI_NOMBRE, WIFI_CLAVE);
  Serial.print("Conectando al WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("Abre en el navegador: http://");
  Serial.println(WiFi.localIP());
  
  // Definir las dos "páginas" del servidor
  servidor.on("/", []() { servidor.send_P(200, "text/html", PAGINA); });
  servidor.on("/predict", enviarPrediccion);
  servidor.begin();
  
  */
  
  // Intentar conectar a tu WiFi durante 15 segundos
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);
  delay(200);
  WiFi.begin(WIFI_NOMBRE, WIFI_CLAVE);
  Serial.print("Conectando al WiFi");
  int intentos = 0;
  while (WiFi.status() != WL_CONNECTED && intentos < 30) {
    delay(500);
    Serial.print(".");
    intentos++;
  }
  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("Abre en el navegador: http://");
    Serial.println(WiFi.localIP());
  } else {
    // Si falla, la placa crea su propia red
    Serial.println("No se pudo conectar. Creando red propia...");
    Serial.println("Redes que ve el ESP32:");
    int n = WiFi.scanNetworks();
    for (int i = 0; i < n; i++) Serial.println(WiFi.SSID(i));

    WiFi.mode(WIFI_AP);
    WiFi.softAP("Varita", "12345678");
    Serial.println("Conectate a la red 'Varita' (clave 12345678)");
    Serial.print("y abre: http://");
    Serial.println(WiFi.softAPIP());   // normalmente 192.168.4.1
  }

  // Definir las dos "páginas" del servidor
  servidor.on("/", []() { servidor.send_P(200, "text/html", PAGINA); });
  servidor.on("/predict", enviarPrediccion);
  servidor.begin();
  Serial.println("Servidor web iniciado");

}
void loop() {
  servidor.handleClient();   // atiende al navegador

  static float gesto[N_OUT][6];
  if (gestureUpdate(gesto)) {   // true cuando sueltas el botón
    predecir(gesto);
  }
}