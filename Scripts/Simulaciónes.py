import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parámetros del MESFET
# =========================
IDSS = 10e-3   # Corriente IDSS [A]
VP = -3.0     # Tensión de pinch-off [V]

# =========================
# 1) CURVAS DE SALIDA ID(VDS)
# =========================
VGS_values = [0.0, -0.5, -1.0, -1.5, -2.0, -2.5, -3.0]
VDS = np.linspace(0, 6, 400)

plt.figure()

for VGS in VGS_values:
    ID = np.zeros_like(VDS)

    for i, vds in enumerate(VDS):

        # Modo corte
        if VGS <= VP:
            ID[i] = 0.0

        # Modo óhmico
        elif vds < (VGS - VP):
            ID[i] = (2 * IDSS / VP**2) * (
                (VGS - VP) * vds - 0.5 * vds**2
            )

        # Modo estrangulación
        else:
            ID[i] = IDSS * (1 - VGS / VP)**2

    plt.plot(VDS, ID, label=f"$V_{{GS}}$ = {VGS:.1f} V")

plt.xlabel("$V_{DS}$ [V]")
plt.ylabel("$I_D$ [A]")
plt.title("Curvas de salida $I_D$–$V_{DS}$ (MESFET n-canal)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# 2) CURVA DE TRANSFERENCIA ID(VGS)
# =========================
VGS = np.linspace(VP - 0.5, 0.2, 400)
ID = np.zeros_like(VGS)

for i, vgs in enumerate(VGS):

    # Modo corte
    if vgs <= VP:
        ID[i] = 0.0

    # Modo estrangulación
    else:
        ID[i] = IDSS * (1 - vgs / VP)**2

plt.figure()
plt.plot(VGS, ID, linewidth = 4)
plt.xlabel("$V_{GS}$ [V]")
plt.ylabel("$I_D$ [A]")
plt.title("Curva de transferencia $I_D$–$V_{GS}$ (MESFET n-canal)")
plt.grid(True)
plt.tight_layout()
plt.show()
