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
Nd = 4e15                     # cm^-3


#Geometricos
a = 120e-6               # cm
Z = 1000e-6                     # cm
L = 200e-6                      # cm

phi_M = 4.33 
chi_GaAs = 4.07
V_bi = phi_M - chi_GaAs                    # V
W_d0 = np.sqrt(2*(eps_s*V_bi/(q*Nd)))

# Tensiones de control
V_P  =  -(q*Nd*(a**2))/(2*eps_s)       # V  (pinch-off)

V_GS = 0

V_DS_SAT  =  (-V_P + V_GS - V_bi )


IDSS = (Z/L) * (a - W_d0 ) * q * mu_n * Nd * V_DS_SAT



print("V_bi = "+ str(V_bi)+ "V")
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


###################################################################################################################

#Modelo completo
#VGS_completo = np.linspace(V_P - 3, 0, 600)
go = (q*mu_n*Nd*Z*a)/(L)

print("go = "+ str(go) + "1/ohm")

VGS = np.linspace(V_P, 0, 1000)

VDS_sat = (-V_P + VGS - V_bi )

arg1 = VDS_sat + V_bi - VGS
arg2 = V_bi - VGS

termino = ((2)/(3*np.sqrt(np.abs(V_P))))*( (arg1)**(3/2) - (arg2)**(3/2) )

ID_completo = go * ( VDS_sat -termino ) 


#Modelo 2


ID_norm = (Z * mu_n * (q**2) * (Nd**2) * (a**3)) / (6 * eps_s * L)
termino_2 = ((VDS_sat + VGS + V_bi)**(3/2) - (VGS - V_bi)**(3/2))

ID_model_2 = ID_norm*(3*VDS_sat/V_P - (2/(V_P**(3/2)))*termino_2 )

plt.figure()
plt.plot(VGS_corte, ID_corte, linewidth=4, label="Corte")
plt.plot(VGS_estr, ID_estr, linewidth=4, label="Estrangulamiento")
plt.plot(VGS, ID_completo, linewidth=4, label="Modelo Completo",color = "green" , linestyle = "--")
plt.plot(VGS, ID_model_2, linewidth=4, label="Modelo Completo 2",color = "red" , linestyle = "--")
plt.plot(VGS_corte, ID_corte, linewidth=4,color = "green" , linestyle = "--")
plt.axvline(V_P, color='gray', linestyle='--', linewidth=2, label=r"$V_P$")
plt.xlabel(r"$V_{GS}$ [V]")
plt.ylabel(r"$I_D$ [A]")
plt.title("Curva de transferencia MESFET")
plt.grid(True)

# ---- Cartel de parámetros ----
label_text = (
    r"MESFET (GaAs / Ti)" "\n"
    rf"$N_D = {Nd:.2e}\ \mathrm{{cm^{{-3}}}}$" "\n"
    rf"$\mu_n = {mu_n:.0f}\ \mathrm{{cm^2/Vs}}$" "\n"
    rf"$a = {a*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$L = {L*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$Z = {Z*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$V_P = {V_P:.1f}\ \mathrm{{V}}$"
)

plt.text(
    0.05, 0.5, label_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
plt.legend()
plt.show()







VGS_vals = [-2.0, V_P, -1.5, -1.0, -0.5, 0.0]

plt.figure()

for VGS in VGS_vals:

    VDS_sat = max(VGS - V_P, 0)
    VDS = np.linspace(0, VDS_sat, 300)

    # Región óhmica
    ID_ohmico = (2 * IDSS / V_P**2) * (
        (VGS - V_P) * VDS - VDS**2 / 2
    )

    # Label
    if np.isclose(VGS, V_P):
        label = r"$V_{GS}=V_P$"
    else:
        label = rf"$V_{{GS}}={VGS}\,\mathrm{{V}}$"

    plt.plot(VDS, ID_ohmico, linewidth=3, label=label)

    # -------------------------
    # IDSAT (región saturación)
    # -------------------------
    if VGS > V_P:
        IDSAT = IDSS * (1 - VGS / V_P)**2

        VDS_sat_line = np.linspace(VDS_sat, max(VGS_vals) - V_P + 0.2, 50)
        ID_sat_line = IDSAT * np.ones_like(VDS_sat_line)

        plt.plot(
            VDS_sat_line,
            ID_sat_line,
            linestyle="--",
            linewidth=3,
            color=plt.gca().lines[-1].get_color()
        )

plt.xlabel(r"$V_{DS}$ [V]")
plt.ylabel(r"$I_D$ [A]")
plt.title("Curva de salida MESFET")
plt.grid(True)
plt.legend()
plt.show()
