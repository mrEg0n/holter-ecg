# Copy this file to wifi_config.py and fill in your values.
# wifi_config.py is gitignored so credentials never reach GitHub.
#
# NETWORKS è una lista di reti che il Pico proverà a contattare.
# Per ognuna serve l'IP del server (Mac/Pi 5) sulla LAN locale —
# in genere diverso per ciascuna rete.

NETWORKS = [
    {
        "ssid":      "YOUR_WIFI_SSID",
        "password":  "YOUR_WIFI_PASSWORD",
        "server_ip": "192.168.x.x",
    },
    # aggiungi qui altre reti se il dispositivo si sposta:
    # {"ssid": "...", "password": "...", "server_ip": "..."},
]

SERVER_PORT = 5005

SSID        = NETWORKS[0]["ssid"]
PASSWORD    = NETWORKS[0]["password"]
SERVER_IP   = NETWORKS[0]["server_ip"]
