#include "gesture_core.h"

void setup() {
  Serial.begin(115200);
  gestureInit();
}

void loop() {
  static float out[N_OUT][6];
  if (gestureUpdate(out)) {
    Serial.print("G");
    for (int i = 0; i < N_OUT; i++)
      for (int k = 0; k < 6; k++) { Serial.print(','); Serial.print(out[i][k], 4); }
    Serial.println();
  }
}