"""
Plot del file CSV registrato dal Pico.
Mostra il tracciato ECG su 5 secondi.
"""
import csv
import sys
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "ecg_take1.csv"

t_ms = []
volt = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            t_ms.append(float(parts[0]))
            volt.append(float(parts[2]))
        except ValueError:
            continue

print(f"loaded {len(volt)} samples")
print(f"voltage range: {min(volt):.3f} - {max(volt):.3f} V")
print(f"swing peak-to-peak: {(max(volt)-min(volt))*1000:.0f} mV")

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot([t/1000 for t in t_ms], volt, linewidth=0.8, color='#c0392b')
ax.set_xlabel("Time (s)")
ax.set_ylabel("AD8232 output (V)")
ax.set_title("ECG raw — Pico 2 W + AD8232 — 250 Hz")
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 3.3)
plt.tight_layout()
out = path.replace(".csv", ".png")
plt.savefig(out, dpi=120)
print(f"saved plot to {out}")
plt.show()
