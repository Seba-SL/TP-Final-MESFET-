import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Eje espacial
# ----------------------------
y_metal = np.linspace(-3, 0, 100)   # Metal
y_sc = np.linspace(0, 3, 200)       # Semiconductor

# ----------------------------
# Parámetros energéticos (ejemplo)
# ----------------------------
Ef = 1.0          # Nivel de Fermi (constante)
E0_metal = 2.2    # Nivel de vacío en el metal
E0_sc = 2.4       # Nivel de vacío en el semiconductor

Eg = 1.4          # Gap del semiconductor (GaAs aprox.)

# ----------------------------
# Bandas en el semiconductor
# (curvatura por región de vaciamiento)
# ----------------------------
Ec_sc = 0.8 + 0.25 * y_sc
Ev_sc = Ec_sc - Eg

# ----------------------------
# Bandas en el metal (planas)
# ----------------------------
Ef_metal = Ef * np.ones_like(y_metal)
E0_metal_line = E0_metal * np.ones_like(y_metal)

# ----------------------------
# Gráfica
# ----------------------------
plt.figure(figsize=(8,4))

# Metal
plt.plot(y_metal, Ef_metal, linestyle='--', label=r'$E_f$')
plt.plot(y_metal, E0_metal_line, label=r'$E_0$ (Metal)')

# Semiconductor
plt.plot(y_sc, Ec_sc, label=r'$E_c$')
plt.plot(y_sc, Ev_sc, label=r'$E_v$')
plt.plot(y_sc, E0_sc * np.ones_like(y_sc), linestyle=':', label=r'$E_0$ (SC)')

# Nivel de Fermi extendido
plt.plot([-3, 3], [Ef, Ef], linestyle='--')

# Interfase metal–semiconductor
plt.axvline(0)

# Etiquetas
plt.xlabel(r'Posición $y$')
plt.ylabel(r'Energía')
plt.title('Diagrama de bandas Metal – Semiconductor tipo N (equilibrio)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
