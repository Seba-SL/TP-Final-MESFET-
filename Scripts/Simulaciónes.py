import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# -----------------------------
# Constantes físicas utilizadas
# -----------------------------
q = 1.602e-19                 # C
eps0 = 88.5e-15              # F/cm
eps_r_GaAs = 12.9
eps_s = eps_r_GaAs * eps0     # F/cm
k= 8.617e-5  #eV/K
T = 300  # K
# -----------------------------
# Parámetros del MESFET
# -----------------------------
mu_n = 8500               # 8500 cm^2/Vs
Nd = 4e15                     # cm^-3
ni= 1.79e6                  # cm^-3

#Geometricos
a = 120e-6               # cm
Z = 1000e-6                     # cm  (equivalente a )
L = 400e-6                      # cm


phi_M = 4.33 
chi_GaAs = 4.07
E_g = 1.42   #eV

E_C =E_g 

E_F = E_g/2 + k*T*np.log(Nd/ni)

V_bi = (phi_M - chi_GaAs )   - (E_C - E_F)               # V
W_d0 = np.sqrt(2*(eps_s*V_bi/(q*Nd)))
go = (q*mu_n*Nd*Z*a)/(L)

# Tensiones de control
V_P_0 = (q*Nd*(a**2))/(2*eps_s)
V_P  =  V_bi - V_P_0       # V  (pinch-off)

V_GS = 0

V_DS_SAT  =  V_P_0 + V_GS - V_bi 



############### Modelo Clasico #######################################################################################

## IDSS tomada del modelo completo, se tomaria experimentalmente para modelo clasico
IDSS =go*(V_DS_SAT - (2/(3*np.sqrt(V_P_0) ))*((V_bi + V_DS_SAT)**(3/2) - (V_bi)**(3/2) )        )

# Rango de VGS
VGS_estrangulamiento = np.linspace(V_P , 0, 500)

# Corte
VGS_corte = np.linspace(V_P - 3, V_P, 300)
ID_corte = np.zeros_like(VGS_corte)

# Estrangulamiento (Shockley)
VGS_estr = np.linspace(V_P, 0, 500)
ID_estr = IDSS * (1 - VGS_estr / V_P)**2



############### Modelo Completo #######################################################################################

print("go = "+ str(go) + "1/ohm")

VGS = np.linspace(V_P, 0, 1000)


VDS_sat = (-V_P + VGS - V_bi )

arg1 = VDS_sat + V_bi - VGS
arg2 = V_bi - VGS

termino = ((2)/(3*np.sqrt(np.abs(V_P))))*( (arg1)**(3/2) - (arg2)**(3/2) )

ID_completo = go * ( VDS_sat -termino ) 




############### Curvas de Transferencia ################################################################################

plt.figure()
plt.plot(VGS_corte, ID_corte*1e3, linewidth=4, label="Corte")
plt.plot(VGS_estr, ID_estr*1e3, linewidth=4, label="Estrangulamiento")
plt.plot(VGS, ID_completo*1e3, linewidth=4, label="Modelo Completo",color = "green" , linestyle = "--")
plt.plot(VGS_corte, ID_corte*1e3, linewidth=4,color = "green" , linestyle = "--")
plt.axvline(V_P, color='gray', linestyle='--', linewidth=2, label=r"$V_P$")
plt.xlabel(r"$V_{GS}$ [V]")
plt.ylabel(r"$I_D$ [mA]")
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
    rf"$V_p = {V_P:.1f}\ \mathrm{{V}}$""\n"
    rf"$IDSS = {IDSS*1e3:.1f}\ \mathrm{{mA}}$"
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





############### Curvas de Salida ################################################################################


VGS_vals = [-3.9,-3.5,-3, -2.0, -1.5, -1.0, -0.5, 0.0]

plt.figure()

for VGS in VGS_vals:

    VDS_sat = max(VGS - V_P, 0)
    VDS = np.linspace(0, VDS_sat, 300)

    # -------------------------
    # MODELO SHOCKLEY
    # -------------------------
    ID_ohmico = (2 * IDSS / (V_P**2)) * (
        (VGS - V_P) * VDS - VDS**2 / 2
    )

    if np.isclose(VGS, V_P):
        label = r"$V_{GS}=V_P$"
    else:
        label = rf"$V_{{GS}}={VGS}\,\mathrm{{V}}$"

    # Dibujamos Shockley y guardamos color
    line, = plt.plot(VDS, ID_ohmico*1e3, linewidth=3, label=label)

    color_actual = line.get_color()

    # Saturación Shockley
    if VGS > V_P:
        IDSAT = IDSS * (1 - VGS / V_P)**2
        VDS_sat_line = np.linspace(VDS_sat, 10, 300)
        ID_sat_line = IDSAT * np.ones_like(VDS_sat_line)

        plt.plot(VDS_sat_line, ID_sat_line*1e3,
                 linewidth=3, color=color_actual)

    # -------------------------
    # MODELO COMPLETO
    # -------------------------
    factor = (2/(3*np.sqrt(V_P_0)))

    ID_completo = go*(VDS - factor * (
        (V_bi - VGS + VDS)**(3/2) -
        (V_bi - VGS)**(3/2)
    ))

    plt.plot(VDS, ID_completo*1e3,linewidth=3,linestyle="--", color=color_actual)

    # Saturación modelo completo
    if VGS > V_P:

        VDS_sat = V_P_0 + VGS - V_bi

        ID_sat_completo = go*(VDS_sat - factor * ((V_bi - VGS + VDS_sat)**(3/2)- (V_bi - VGS)**(3/2)))

        ID_sat_line = ID_sat_completo * np.ones_like(VDS_sat_line)

        plt.plot(VDS_sat_line, ID_sat_line*1e3,linewidth=3,linestyle="--",color=color_actual)
        # -------------------------

label_text = (
    r"MESFET (GaAs / Ti)" "\n"
    r"────────  Modelo clásico" "\n"
    r"- - - - -  Modelo completo" "\n"
    "\n"
    rf"$N_D = {Nd:.2e}\ \mathrm{{cm^{{-3}}}}$" "\n"
    rf"$\mu_n = {mu_n:.0f}\ \mathrm{{cm^2/Vs}}$" "\n"
    rf"$a = {a*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$L = {L*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$Z = {Z*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$V_p = {V_P:.1f}\ \mathrm{{V}}$" "\n"
    rf"$IDSS = {IDSS*1e3:.1f}\ \mathrm{{mA}}$"
)


plt.text(
    0.75, 0.75, label_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)
# -------------------------
plt.xlabel(r"$V_{DS}$ [V]")
plt.ylabel(r"$I_D$ [mA]")
plt.title("Curva de salida MESFET")
plt.grid(True)
plt.legend()
plt.show()




################################################################################################################################################################################

# Modelo Completo + Efectos no ideales 

################################- Modulación  del Canal (Sobre curva de salida) #########################################################################################
VGS_vals = [-1.5]

VDS_sat = max(VGS - V_P, 0)
VDS = np.linspace(0.01, VDS_sat, 300)

factor = (2/(3*np.sqrt(V_P_0)))

ID_completo = go*(VDS - factor * ( (V_bi - VGS + VDS)**(3/2) - (V_bi - VGS)**(3/2) ))

plt.plot(VDS, ID_completo*1e3,linewidth=3,linestyle="-",alpha = 0.9, color="red")



# Saturación modelo completo
if VGS > V_P:

        
        ID_sat_completo = go*(VDS_sat - factor * ((V_bi - VGS + VDS_sat)**(3/2)- (V_bi - VGS)**(3/2)))

        VDS_sat_line = np.linspace(VDS_sat, 10, 1000)

        Delta_L = np.sqrt(2*eps_s*(VDS_sat_line - VDS_sat)/(q*Nd))


        L_prima = L - 0.5*Delta_L

        ID_sat_line = ID_sat_completo * np.ones_like(VDS_sat_line)

        
        ID_sat_line_ch_modulation =ID_sat_line*(L/L_prima)


        L_prima = L*5 - 0.5*Delta_L

        ID_sat_line_ch_modulation_B = ID_sat_line*(5*L/L_prima)

        plt.plot(VDS_sat_line, ID_sat_line*1e3,linewidth=3.5,linestyle="--", alpha = 0.6,color="blue")
        plt.plot(VDS_sat_line, ID_sat_line_ch_modulation*1e3,linewidth=3.5,linestyle="-", alpha = 0.6,color="red")
        plt.plot(VDS_sat_line, ID_sat_line_ch_modulation_B*1e3,linewidth=3.5,linestyle="-", alpha = 0.6,color="orange")

      
        # -------------------------

label_text = (
    r"MESFET (GaAs / Ti)" "\n"
    r"──────── Modelo con efecto de modulación" "\n"
    r"- - - - -  Modelo completo sin efecto" "\n"
    "\n"
    rf"$N_D = {Nd:.2e}\ \mathrm{{cm^{{-3}}}}$" "\n"
    rf"$\mu_n = {mu_n:.0f}\ \mathrm{{cm^2/Vs}}$" "\n"
    rf"$a = {a*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$L = {L*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$Z = {Z*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$V_p = {V_P:.1f}\ \mathrm{{V}}$" "\n"
    rf"$IDSS = {IDSS*1e3:.1f}\ \mathrm{{mA}}$""\n"
)

plt.text(0.6, 0.5, label_text,transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))


plt.text(0.04, 0.7, r"Sin modulación",transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="blue", alpha=0.5))
plt.text(0.04, 0.8, r"Modulación para canal largo (L = 20 $\mu m$)",transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="orange", alpha=0.5))
plt.text(0.04, 0.9, r"Modulación para canal corto (L = 4 $\mu m$)",transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))
# -------------------------
# -------------------------
plt.xlabel(r"$V_{DS}$ [V]")
plt.ylabel(r"$I_D$ [mA]")
plt.title("Curva de salida MESFET (con efecto de modulación del canal)")
plt.grid(True)
plt.legend()
plt.show()



################################# Saturación de la velocidad de arrastre #########################################################################################

campo_critico = 1e5

VDS = np.linspace(0.01, VDS_sat*10, 1000)
Delta_L = np.sqrt(np.abs(2*eps_s*(VDS - VDS_sat)/(q*Nd)))

L_prima = L- 0.5*Delta_L
vsat =  6.8e6  # cm/s

print("Valor de vsat "+ str(vsat) )
# Parámetros de ajuste del modelo
xi_p = 3e3           # V/cm  (posición del pico)
xi_s = 2e4           # V/cm  (campo donde se estabiliza vsat)

# --------------------------------------------------
# Rango de campo eléctrico (logarítmico)
# --------------------------------------------------
xi = VDS/L_prima      

# --------------------------------------------------
# Modelo con pico
# --------------------------------------------------
v_arr =  (mu_n * xi / (1 + (xi/xi_p)**2) + vsat*((xi/xi_s)**2) / (1 + (xi/xi_s)**2))

# --------------------------------------------------
# Gráfico
# --------------------------------------------------
plt.figure(figsize=(7,5))
plt.plot(xi, v_arr, linewidth = 4 )

xi_ohmica_max = 2e3
xi_ndm_max = 1.5e4

# Región óhmica
plt.axvspan(xi.min(), xi_ohmica_max, 
            color="blue", alpha=0.08)

# Región NDM
plt.axvspan(xi_ohmica_max, xi_ndm_max, 
            color="red", alpha=0.08)

# Región saturación
plt.axvspan(xi_ndm_max, xi.max()*5, color="green", alpha=0.08)


# Textos en cada región
plt.text(4e1, 1e7, "Región de movilidad constante", fontsize=15)
plt.text(3e3, 2e6, "Movilidad diferencial\nnegativa (NDM)", fontsize=15)
plt.text(4e4, 2e6, "Saturación\nde velocidad", fontsize=15)


plt.text(xi_p, 1e6,r'$\xi_p$', fontsize=15)
plt.text(xi_s, 1e6, r'$\xi_s$', fontsize=15)

plt.axvline(xi_p, linestyle='--', linewidth=2)
plt.axvline(xi_s, linestyle='--', linewidth=2)

plt.text(xi_p, plt.ylim()[0]*2, r'$\xi_p$', rotation=90)
plt.text(xi_s, plt.ylim()[0]*2, r'$\xi_s$', rotation=90)


# Obtener ticks actuales
ax = plt.gca()
ticks = ax.get_xticks()

# Añadir los característicos
special_ticks = [xi_p, xi_s]
all_ticks = sorted(list(set(ticks.tolist() + special_ticks)))

ax.set_xticks(all_ticks)

# Formateador personalizado
def log_formatter(x, pos):
    if np.isclose(x, xi_p):
        return r'$\xi_p$'
    elif np.isclose(x, xi_s):
        return r'$\xi_s$'
    else:
        return ticker.LogFormatterMathtext()(x)

ax.xaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Campo eléctrico ξ [V/cm]")
plt.ylabel("Velocidad de arrastre v [cm/s]")
plt.title("Velocidad de arrastre en GaAs ")
plt.grid(True, which="both")

plt.show()


################### Curva de salida con efectos no ideales 1, y 2
VGS_vals = [-1.5]

VDS_sat = max(VGS - V_P, 0)
VDS = np.linspace(0.01, VDS_sat, 1000)

factor = (2/(3*np.sqrt(V_P_0)))

ID_completo = go*(VDS - factor * ( (V_bi - VGS + VDS)**(3/2) - (V_bi - VGS)**(3/2) ))

plt.plot(VDS, ID_completo*1e3,linewidth=3,linestyle="-",alpha = 0.9, color="red")

# Saturación modelo completo
if VGS > V_P:

        
        ID_sat_completo = go*(VDS_sat - factor * ((V_bi - VGS + VDS_sat)**(3/2)- (V_bi - VGS)**(3/2)))

        VDS_sat_line = np.linspace(VDS_sat, 10, 1000)
        Delta_L = np.sqrt(2*eps_s*(VDS_sat_line - VDS_sat)/(q*Nd))


        L_prima = L - 0.5*Delta_L

        ID_sat_line = ID_sat_completo * np.ones_like(VDS_sat_line)

        
        ID_sat_line_ch_modulation =ID_sat_line*(L/L_prima)*(vsat/(campo_critico*mu_n))

        plt.plot(VDS_sat_line, ID_sat_line*1e3,linewidth=3.5,linestyle="--", alpha = 0.6,color="blue")
        plt.plot(VDS_sat_line, ID_sat_line_ch_modulation*1e3,linewidth=3.5,linestyle="-", alpha = 0.6,color="red")
       
      
        # -------------------------

label_text = (
    r"MESFET (GaAs / Ti)" "\n"
    r"──────── Modelo con efecto de modulación" "\n"
    r"- - - - -  Modelo completo sin efecto" "\n"
    "\n"
    rf"$N_D = {Nd:.2e}\ \mathrm{{cm^{{-3}}}}$" "\n"
    rf"$\mu_n = {mu_n:.0f}\ \mathrm{{cm^2/Vs}}$" "\n"
    rf"$a = {a*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$L = {L*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$Z = {Z*1e4:.1f}\ \mu\mathrm{{m}}$" "\n"
    rf"$V_p = {V_P:.1f}\ \mathrm{{V}}$" "\n"
    rf"$IDSS = {IDSS*1e3:.1f}\ \mathrm{{mA}}$""\n"
)

#plt.text(0.6, 0.5, label_text,transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))




plt.text(0.04, 0.7, r"Sin modulación",transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="blue", alpha=0.5))
#plt.text(0.04, 0.8, r"Modulación para canal largo (L = 20 $\mu m$)",transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="orange", alpha=0.5))
plt.text(0.04, 0.9, r"Modulación para canal corto (L = 4 $\mu m$)",transform=plt.gca().transAxes,fontsize=10,verticalalignment='top',bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))
# -------------------------
# -------------------------
plt.xlabel(r"$V_{DS}$ [V]")
plt.ylabel(r"$I_D$ [mA]")
plt.title(r"Curva de salida MESFET (con afectada por la saturación de $v_{arr}$ )")
plt.grid(True)
plt.legend()
plt.show()

