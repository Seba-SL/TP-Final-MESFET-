import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Constantes físicas
# -----------------------------
q = 1.602e-19                 # C
eps0 = 88.5e-15              # F/cm
eps_r_GaAs = 12.9
eps_s = eps_r_GaAs * eps0     # F/cm

# -----------------------------
# Parámetros del MESFET
# -----------------------------
mu_n = 8500               # 8500 cm^2/Vs
Nd = 2e15                     # cm^-3


#Geometricos
a = 120e-6               # cm
Z = 1000e-6                     # cm
L = 200e-6                      # cm
V_bi = 0.6                    # V
W_d0 = np.sqrt(2*(eps_s*V_bi/(q*Nd)))

# Tensiones de control
V_P  =  -(q*Nd*(a**2))/(2*eps_s)         # V  (pinch-off)

V_GS = 0

V_DS_SAT  = V_bi - V_P  -V_GS 


IDSS = (Z/L) * (a - W_d0) * q * mu_n * Nd * V_DS_SAT


print("W_d0 = "+ str(W_d0)+ "cm \n")

print("Vp = "+ str(V_P) + "V")

print("VDS_[sat] = "+ str(V_DS_SAT ) + "V")



print("I_DSS = "+ str(-IDSS) +"A")

# Rango de VGS
VGS_estrangulamiento = np.linspace(V_P , 0, 500)





# Corte
VGS_corte = np.linspace(V_P - 3, V_P, 300)
ID_corte = np.zeros_like(VGS_corte)

# Estrangulamiento (Shockley)
VGS_estr = np.linspace(V_P, 0, 500)
ID_estr = IDSS * (1 - VGS_estr / V_P)**2

#Modelo completo
#VGS_completo = np.linspace(V_P - 3, 0, 600)
go = (q*mu_n*Nd*Z*a)/(L)

print("go = "+ str(go) + "1/ohm")

VGS = np.linspace(V_P, 0, 500)

VDS_sat = V_bi - V_P  - VGS 

arg1 = (VDS_sat + V_bi - VGS)
arg2 = (V_bi - VGS)

termino = (2/3) * (arg1**(3/2) - arg2**(3/2)) / np.sqrt(-V_P)

ID_completo = go * (VDS_sat - termino)

plt.figure()
plt.plot(VGS_corte, ID_corte, linewidth=3, label="Corte")
plt.plot(VGS_estr, ID_estr, linewidth=3, label="Estrangulamiento")
plt.plot(VGS, ID_completo, linewidth=3, label="Estrangulamiento",color = "green" , linestyle = "--")
plt.xlabel(r"$V_{GS}$ [V]")
plt.ylabel(r"$I_D$ [A]")
plt.title("Característica de transferencia JFET")
plt.grid(True)
plt.legend()
plt.show()

# =========================
# REGION OHMICA ID vs VDS
# =========================

VGS_vals = [-2.0, V_P,-1.5,-1.0, -0.5, 0.0]

plt.figure()

for VGS in VGS_vals:
    VDS_max = max(VGS - V_P, 0)
    VDS = np.linspace(0, VDS_max, 300)

    ID_ohmico = (2 * IDSS / V_P**2) * (
        (VGS - V_P) * VDS - VDS**2 / 2
    )

    plt.plot(VDS, ID_ohmico, label=rf"$V_{{GS}}={VGS}$ V")

plt.xlabel(r"$V_{DS}$ [V]")
plt.ylabel(r"$I_D$ [A]")
plt.title("Región óhmica del JFET")
plt.grid(True)
plt.legend()
plt.show()
