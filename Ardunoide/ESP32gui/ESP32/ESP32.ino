#include <WiFi.h>
#include <esp_now.h>
#include <esp_wifi.h>

#define ESPNOW_CHANNEL 1

// ======================================================
// ROBOT ID 8
// ======================================================
uint8_t robot8Mac[] = {
    0x58, 0x2A, 0xBD, 0x77, 0x1A, 0x58
};

// ======================================================
// ROBOT ID 9
// ======================================================
uint8_t robot9Mac[] = {
    0x1C, 0x69, 0x20, 0xA4, 0xD0, 0x58
};

// ======================================================
// ROBOT ID 3
//
// CHUA CO MAC THU 3.
// THAY 6 BYTE BEN DUOI BANG MAC ESP32 GAN ROBOT ID 3.
// ======================================================
uint8_t robot3Mac[] = {
    0x40, 0x22, 0xD8, 0x4F, 0x07, 0xE0
};


// Python gui:
//   8;START#
//   8;STOP#
//   8;352.4;64.7;43.2#
//
// Danh sach diem dich trong MOT packet, KHONG co so luong:
//   8;WPLIST;100.0;80.0;120.0;60.0;140.0;80.0#
//
// Xoa diem dich:
//   8;WPCLR#
//
//   9;START#
//   9;STOP#
//   9;420.8;80.1;271.6#
//
//   3;START#
//   3;STOP#
//   3;500.2;40.0;90.0#
//
// Gateway doc ID dau packet, chon dung MAC,
// sau do BO ID va gui phan con lai cho ESP32 tren robot.
//
// Vi du:
// Python:  8;352.4;64.7;43.2#
// ESP-NOW: 352.4;64.7;43.2# -> robot ID 8
// ======================================================

char serialBuffer[320];
int serialIndex = 0;


void printMac(const uint8_t *mac)
{
    for (int i = 0; i < 6; i++)
    {
        if (mac[i] < 0x10)
            Serial.print("0");

        Serial.print(mac[i], HEX);

        if (i < 5)
            Serial.print(":");
    }

    Serial.println();
}


bool isZeroMac(const uint8_t *mac)
{
    for (int i = 0; i < 6; i++)
    {
        if (mac[i] != 0x00)
            return false;
    }

    return true;
}


bool addPeer(const uint8_t *mac, const char *label)
{
    if (isZeroMac(mac))
    {
        Serial.print(">>> CHUA KHAI BAO MAC ");
        Serial.println(label);
        return false;
    }

    if (esp_now_is_peer_exist(mac))
        esp_now_del_peer(mac);

    esp_now_peer_info_t peerInfo = {};

    memcpy(peerInfo.peer_addr, mac, 6);
    peerInfo.channel = ESPNOW_CHANNEL;
    peerInfo.ifidx = WIFI_IF_STA;
    peerInfo.encrypt = false;

    esp_err_t result = esp_now_add_peer(&peerInfo);

    if (result != ESP_OK)
    {
        Serial.print(">>> ADD PEER FAILED ");
        Serial.print(label);
        Serial.print(": ");
        Serial.println((int)result);
        return false;
    }

    Serial.print(">>> ADD PEER OK ");
    Serial.print(label);
    Serial.print(": ");
    printMac(mac);

    return true;
}


const uint8_t* getRobotMac(int robotId)
{
    if (robotId == 8)
        return robot8Mac;

    if (robotId == 9)
        return robot9Mac;

    if (robotId == 3)
        return robot3Mac;

    return nullptr;
}


bool sendToRobot(
    int robotId,
    const char *payload,
    int payloadLength
)
{
    const uint8_t *targetMac = getRobotMac(robotId);

    if (targetMac == nullptr)
    {
        Serial.print(">>> ROBOT ID KHONG HO TRO: ");
        Serial.println(robotId);
        return false;
    }

    if (isZeroMac(targetMac))
    {
        Serial.print(">>> CHUA CO MAC CHO ROBOT ID ");
        Serial.println(robotId);
        return false;
    }

    // ESP-NOW v1.0: payload toi da 250 byte.
    // Python dang gioi han WPLIST <= 240 byte de co du bien an toan.
    if (payloadLength > 250)
    {
        Serial.print(">>> PAYLOAD QUA DAI: ");
        Serial.println(payloadLength);
        return false;
    }

    esp_err_t result = esp_now_send(
        targetMac,
        (const uint8_t *)payload,
        payloadLength
    );

    if (result != ESP_OK)
    {
        Serial.print(">>> ESP-NOW SEND ERROR R");
        Serial.print(robotId);
        Serial.print(": ");
        Serial.println((int)result);
        return false;
    }

    return true;
}


void processSerialPacket(char *packet)
{
    // Packet co dang:
    // 8;START#
    // 8;STOP#
    // 8;352.4;64.7;43.2#

    char *separator = strchr(packet, ';');

    if (separator == nullptr)
    {
        Serial.print(">>> PACKET SAI FORMAT: ");
        Serial.println(packet);
        return;
    }

    // Cat chuoi tai dau ';' dau tien.
    *separator = '\0';

    int robotId = atoi(packet);

    // payload bat dau sau dau ';'
    char *payload = separator + 1;

    int payloadLength = strlen(payload);

    if (payloadLength <= 0)
        return;

    bool ok = sendToRobot(
        robotId,
        payload,
        payloadLength
    );

    if (!ok)
    {
        Serial.print(">>> KHONG GUI DUOC R");
        Serial.print(robotId);
        Serial.print(" -> ");
        Serial.println(payload);
    }
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

    if (esp_now_init() != ESP_OK)
    {
        Serial.println(">>> ESP-NOW INIT FAILED");

        while (true)
            delay(1000);
    }

    addPeer(robot8Mac, "ROBOT 8");
    addPeer(robot9Mac, "ROBOT 9");
    addPeer(robot3Mac, "ROBOT 3");

    Serial.println();
    Serial.println("====================================");
    Serial.println("ESP32 GATEWAY - 3 ROBOT");
    Serial.println("====================================");

    Serial.print("MAC GATEWAY: ");
    Serial.println(WiFi.macAddress());

    Serial.print("R8  -> ");
    printMac(robot8Mac);

    Serial.print("R9  -> ");
    printMac(robot9Mac);

    Serial.print("R3  -> ");
    printMac(robot3Mac);

    Serial.print("CHANNEL: ");
    Serial.println(ESPNOW_CHANNEL);

    Serial.println();
    Serial.println("CHO DU LIEU TU PYTHON...");
}


void loop()
{
    while (Serial.available() > 0)
    {
        char c = (char)Serial.read();

        if (c == '\r' || c == '\n')
            continue;

        if (serialIndex >= (int)sizeof(serialBuffer) - 1)
        {
            serialIndex = 0;
        }

        serialBuffer[serialIndex++] = c;

        if (c == '#')
        {
            serialBuffer[serialIndex] = '\0';

            processSerialPacket(
                serialBuffer
            );

            serialIndex = 0;
        }
    }
}