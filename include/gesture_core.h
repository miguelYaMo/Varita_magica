#pragma once
#include <Arduino.h>
#include <Wire.h>

#define MPU_ADDR   0x68
#define BTN_PIN    4
#define SDA_PIN    21
#define SCL_PIN    22
#define MAX_S      300
#define N_OUT      64
#define PERIOD_US  10000   // 100 Hz

static float buf_[MAX_S][6];
static int   n_ = 0;
static bool  rec_ = false;
static uint32_t lastUs_ = 0;

inline void mpuInit() {
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x6B); Wire.write(0);    Wire.endTransmission();
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x1C); Wire.write(0x08); Wire.endTransmission(); // ±4g
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x1B); Wire.write(0x08); Wire.endTransmission(); // ±500°/s
}
inline bool readMPU(float *o) {
  Wire.beginTransmission(MPU_ADDR); Wire.write(0x3B); Wire.endTransmission(false);
  if (Wire.requestFrom((uint16_t)MPU_ADDR, (size_t)14, true) < 14) return false;
  int16_t v[7];
  for (int i = 0; i < 7; i++) {
    int16_t alto = Wire.read();   // primero el byte alto
    int16_t bajo = Wire.read();   // luego el byte bajo
    v[i] = (alto << 8) | bajo;
  }
  o[0] = v[0] / 8192.0f; o[1] = v[1] / 8192.0f; o[2] = v[2] / 8192.0f;
  o[3] = v[4] / 65.5f;   o[4] = v[5] / 65.5f;   o[5] = v[6] / 65.5f;
  return true;
}
inline void resample_(float out[N_OUT][6]) {
  for (int i = 0; i < N_OUT; i++) {
    float pos = i * (n_ - 1) / (float)(N_OUT - 1);
    int a = (int)pos, b = min(a + 1, n_ - 1);
    float f = pos - a;
    for (int k = 0; k < 6; k++) out[i][k] = buf_[a][k] * (1 - f) + buf_[b][k] * f;
  }
}
inline void gestureInit() {
  pinMode(BTN_PIN, INPUT_PULLUP);
  mpuInit();
}

// Llamar en cada loop(). Devuelve true cuando se suelta el botón y 'out' tiene un gesto listo.
inline bool gestureUpdate(float out[N_OUT][6]) {
  bool pressed = digitalRead(BTN_PIN) == LOW;

  if (pressed && !rec_) { rec_ = true; n_ = 0; lastUs_ = micros(); }

  if (rec_ && pressed && n_ < MAX_S && micros() - lastUs_ >= PERIOD_US) {
    lastUs_ += PERIOD_US;
    if (readMPU(buf_[n_])) n_++;
  }

  if (rec_ && !pressed) {
    rec_ = false;
    if (n_ >= 20) { resample_(out); return true; }
  }
  return false;
}