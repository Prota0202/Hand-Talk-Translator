/* =============================================================================
 *  Sign Language Glove — ESP32 firmware (3-flex variant)
 *
 *  Acquires:
 *    - 3 flex sensors  (thumb / index / middle, voltage divider with 10k pull-down)
 *    - 1 MPU6050       (accelerometer + gyroscope, I2C @ 0x68)
 *
 *  Sends a single-line CSV record at SAMPLE_HZ over USB serial (115200 baud):
 *
 *      t_ms,f1,f2,f3,ax,ay,az,gx,gy,gz\n
 *
 *  Where:
 *    t_ms : monotonic timestamp (millis())
 *    fN   : raw 12-bit ADC reading (0..4095) for each flex sensor
 *    ax/y/z : acceleration in g    (~+-2g range)
 *    gx/y/z : angular velocity in dps
 *
 *  Pinout
 *  ------
 *    Flex1 (thumb)  -> GPIO 33 (ADC1_CH5)
 *    Flex2 (index)  -> GPIO 32 (ADC1_CH4)
 *    Flex3 (middle) -> GPIO 35 (ADC1_CH7)
 *    MPU6050 SDA    -> GPIO 21
 *    MPU6050 SCL    -> GPIO 22
 *    MPU6050 VCC    -> 3V3
 *    MPU6050 GND    -> GND
 *
 *  Library required (Arduino IDE -> Library Manager):
 *    - MPU6050_light (rfetick)
 * ============================================================================= */

#include <Wire.h>
#include <MPU6050_light.h>

// ── Configuration ─────────────────────────────────────────────────────────────
static const uint8_t  FLEX_PINS[3]    = {33, 32, 35};   // thumb, index, middle
static const uint32_t SAMPLE_HZ       = 50;
static const uint32_t SAMPLE_PERIOD_MS = 1000 / SAMPLE_HZ;
static const uint32_t LED_PIN         = 2;
static const uint32_t SERIAL_BAUD     = 115200;

// ── State ────────────────────────────────────────────────────────────────────
MPU6050   mpu(Wire);
bool      mpu_ready  = false;
uint32_t  next_tick  = 0;
uint32_t  last_blink = 0;
bool      led_on     = false;

// ── Setup ────────────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 2000) { /* wait briefly for USB CDC */ }

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);

  analogReadResolution(12);
  for (uint8_t i = 0; i < 3; i++) {
    analogSetPinAttenuation(FLEX_PINS[i], ADC_11db);
  }

  Wire.begin(21, 22);
  delay(200);

  byte status = mpu.begin();
  if (status == 0) {
    mpu_ready = true;
    Serial.println("# MPU6050 OK, calibration (ne bouge pas 1s)...");
    delay(1000);
    mpu.calcOffsets();
    Serial.println("# Calibration done");
  } else {
    Serial.print("# WARN: MPU6050 not found, status=");
    Serial.println(status);
    Serial.println("# Continuing in flex-only mode (IMU=0)");
  }

  Serial.println("# Sign Glove ready");
  Serial.println("# fmt: t_ms,f1,f2,f3,ax,ay,az,gx,gy,gz");
  next_tick = millis();
}

// ── Loop ─────────────────────────────────────────────────────────────────────
void loop() {
  uint32_t now = millis();
  if ((int32_t)(now - next_tick) < 0) {
    return;
  }
  next_tick += SAMPLE_PERIOD_MS;

  if (now - last_blink > 500) {
    led_on = !led_on;
    digitalWrite(LED_PIN, led_on ? HIGH : LOW);
    last_blink = now;
  }

  uint16_t flex[3];
  for (uint8_t i = 0; i < 3; i++) {
    flex[i] = analogRead(FLEX_PINS[i]);
  }

  float ax = 0, ay = 0, az = 0, gx = 0, gy = 0, gz = 0;
  if (mpu_ready) {
    mpu.update();
    ax = mpu.getAccX();
    ay = mpu.getAccY();
    az = mpu.getAccZ();
    gx = mpu.getGyroX();
    gy = mpu.getGyroY();
    gz = mpu.getGyroZ();
  }

  char line[140];
  snprintf(line, sizeof(line),
           "%lu,%u,%u,%u,%.4f,%.4f,%.4f,%.3f,%.3f,%.3f",
           (unsigned long)now,
           flex[0], flex[1], flex[2],
           ax, ay, az, gx, gy, gz);
  Serial.println(line);
}
