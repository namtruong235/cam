#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>
#include <math.h>

#define ESPNOW_CHANNEL 1

// =============================================================
// DUNG CUNG FILE NAY CHO CA 2 ESP32 RECEIVER.
//
// Gateway da route theo MAC:
//   MAC 40:22:D8:4F:07:E0 <- du lieu Robot ID 8
//   MAC 40:22:D8:3E:75:04 <- du lieu Robot ID 9
//
// Moi receiver chi nhan:
//   X.X;Y.Y;ANGLE.A#
//
// Vi du:
//   352.4;64.7;43.2#
// =============================================================

volatile float robotX = 0.0f;
volatile float robotY = 0.0f;
volatile float robotAngle = 0.0f;

volatile bool newData = false;
volatile bool trackingEnabled = false;
volatile unsigned long lastReceiveTime = 0;
volatile bool waitingPrinted = false;

const unsigned long DATA_TIMEOUT = 300;

void OnDataRecv(
    const esp_now_recv_info_t *info,
    const uint8_t *incomingData,
    int len
)
{
    char buffer[100];

    if (len >= (int)sizeof(buffer))
    {
        len = sizeof(buffer) - 1;
    }

    memcpy(buffer, incomingData, len);
    buffer[len] = '\0';

    // =========================================================
    // START
    // =========================================================
    if (strcmp(buffer, "START#") == 0)
    {
        trackingEnabled = true;
        newData = false;
        lastReceiveTime = millis();
        waitingPrinted = false;

        Serial.println(">>> START");
        return;
    }

    // =========================================================
    // STOP
    // =========================================================
    if (strcmp(buffer, "STOP#") == 0)
    {
        trackingEnabled = false;
        newData = false;
        waitingPrinted = false;

        Serial.println(">>> STOP");
        return;
    }

    if (!trackingEnabled)
        return;

    // =========================================================
    // DATA FLOAT: X;Y;ANGLE#
    // =========================================================
    float x = 0.0f;
    float y = 0.0f;
    float angle = 0.0f;

    int result = sscanf(
        buffer,
        "%f;%f;%f#",
        &x,
        &y,
        &angle
    );

    if (result != 3)
        return;

    // Chuan hoa angle ve [0, 360)
    angle = fmodf(angle, 360.0f);

    if (angle < 0.0f)
    {
        angle += 360.0f;
    }

    robotX = x;
    robotY = y;
    robotAngle = angle;

    newData = true;
    lastReceiveTime = millis();
    waitingPrinted = false;
}

void setup()
{
    Serial.begin(115200);
    delay(1000);

    WiFi.mode(WIFI_STA);
    delay(100);

    esp_wifi_set_ps(WIFI_PS_NONE);
    WiFi.disconnect();

    if (
        esp_wifi_set_channel(
            ESPNOW_CHANNEL,
            WIFI_SECOND_CHAN_NONE
        ) != ESP_OK
    )
    {
        Serial.println(">>> LOI DAT WIFI CHANNEL");
    }

    Serial.println();
    Serial.println("==============================");
    Serial.println("ESP32 ROBOT RECEIVER - FLOAT");
    Serial.println("==============================");

    Serial.print("MAC CUA BOARD NAY: ");
    Serial.println(WiFi.macAddress());

    Serial.print("CHANNEL: ");
    Serial.println(ESPNOW_CHANNEL);

    if (esp_now_init() != ESP_OK)
    {
        Serial.println(">>> ESP-NOW INIT FAILED");
        while (true) delay(1000);
    }

    if (esp_now_register_recv_cb(OnDataRecv) != ESP_OK)
    {
        Serial.println(">>> REGISTER CALLBACK FAILED");
        while (true) delay(1000);
    }

    Serial.println(">>> ESP-NOW READY");
    Serial.println(">>> CHO LENH START#");
}

void loop()
{
    if (!trackingEnabled)
    {
        delay(10);
        return;
    }

    if (newData)
    {
        newData = false;

        // Copy ra bien local de dung cho dieu khien robot.
        float x = robotX;
        float y = robotY;
        float angle = robotAngle;

        Serial.print("X = ");
        Serial.print(x, 1);

        Serial.print(" | Y = ");
        Serial.print(y, 1);

        Serial.print(" | ANGLE = ");
        Serial.print(angle, 1);

        Serial.println(" deg");

        // =====================================================
        // DAT THUAT TOAN DIEU KHIEN MOTOR CUA ROBOT TAI DAY.
        // x, y, angle DEU LA FLOAT.
        // =====================================================
    }

    if (
        millis() - lastReceiveTime > DATA_TIMEOUT
        && !waitingPrinted
    )
    {
        Serial.println(">>> DANG DOI X, Y, ANGLE...");
        waitingPrinted = true;
    }

    delay(1);
}