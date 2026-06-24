# Copy this file to wifi_config.py and fill in your values.
# wifi_config.py is gitignored so credentials never reach GitHub.
#
# NETWORKS is a list of networks the Pico will try to contact.
# Each one needs the IP of the server (Mac/Pi 5) on the local LAN —
# usually different for each network.

NETWORKS = [
    {
        "ssid":      "YOUR_WIFI_SSID",
        "password":  "YOUR_WIFI_PASSWORD",
        "server_ip": "192.168.x.x",
    },
    # add other networks here if the device moves around:
    # {"ssid": "...", "password": "...", "server_ip": "..."},
]

SERVER_PORT = 5005

SSID        = NETWORKS[0]["ssid"]
PASSWORD    = NETWORKS[0]["password"]
SERVER_IP   = NETWORKS[0]["server_ip"]
