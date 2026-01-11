import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def draw_mesfet_total_width(L=10.0, Z=6.0, h=2.0):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # En esta versión, L es la dimensión total en el eje X
    total_width = L
    substrate_thickness = 0.5
    
    def add_block(x_start, x_end, y_start, y_end, z_start, z_end, color, alpha=0.8):
        vertices = [
            [(x_start, y_start, z_start), (x_end, y_start, z_start), (x_end, y_end, z_start), (x_start, y_end, z_start)],
            [(x_start, y_start, z_end), (x_end, y_start, z_end), (x_end, y_end, z_end), (x_start, y_end, z_end)],
            [(x_start, y_start, z_start), (x_start, y_end, z_start), (x_start, y_end, z_end), (x_start, y_start, z_end)],
            [(x_end, y_start, z_start), (x_end, y_end, z_start), (x_end, y_end, z_end), (x_end, y_start, z_end)],
            [(x_start, y_start, z_start), (x_end, y_start, z_start), (x_end, y_start, z_end), (x_start, y_start, z_end)],
            [(x_start, y_end, z_start), (x_end, y_end, z_start), (x_end, y_end, z_end), (x_start, y_end, z_end)]
        ]
        poly = Poly3DCollection(vertices, facecolors=color, linewidths=0.5, edgecolors='black', alpha=alpha)
        ax.add_collection3d(poly)

    # 1. Cuerpo del transistor (Sustrato y Capa Activa) con ancho L
    add_block(-L/2, L/2, 0, Z, -substrate_thickness, 0, '#EEEEEE') # Sustrato
    add_block(-L/2, L/2, 0, Z, 0, h, 'white', alpha=0.3)           # Capa activa
    add_block(-L/4, L/4, 0, Z, h/2, h, 'violet', alpha=0.3)           # Capa activa


    # 2. Terminales (S, G, D) proporcionales a L
    s_w, g_w, d_w = 0.15 * L, 0.4 * L, 0.15 * L
    
    # Source (S)
    add_block(-L/2, -L/2 + s_w, 0, Z, h, h + 0.1, 'silver')
    # Gate (G) - Centrada
    add_block(-g_w/2, g_w/2, 0, Z, h, h + 0.2, 'gray')
    # Drain (D)
    add_block(L/2 - d_w, L/2, 0, Z, h, h + 0.1, 'silver')

    # --- FLECHAS DE COTA (DIMENSIONAMIENTO) ---
    
    # Cota L: Abarca todo el eje x del rectangulo
    y_off = -1.0
    ax.quiver(-L/2, y_off, h-0.5, L, 0, 0, color='red', arrow_length_ratio=0.05)
    ax.quiver(L/2, y_off, h-0.5, -L, 0, 0, color='red', arrow_length_ratio=0.05)
    ax.text(0, y_off, h - 0.2, f'$L$', color='red', ha='center', fontsize=11, fontweight='bold')

    # Cota Z: Profundidad
    x_off_z = L/2 + 0.8
    ax.quiver(x_off_z, 0, 0, 0, Z, 0, color='blue', arrow_length_ratio=0.05)
    ax.quiver(x_off_z, Z, 0, 0, -Z, 0, color='blue', arrow_length_ratio=0.05)
    #ax.text(x_off_z + 0.3, Z/2, 0, f'$Z$', color='blue')
    ax.text(x_off_z + 0.5, Z/2+0.5, 0.5, f'$Z$', color='blue')


    # Cota h: Altura
    x_off_h = -L/2 - 0.8
    ax.quiver(x_off_h, 0, 0, 0, 0, h, color='green', arrow_length_ratio=0.1)
    ax.quiver(x_off_h, 0, h, 0, 0, -h, color='green', arrow_length_ratio=0.1)
    ax.text(x_off_h - 0.8, 0, h/2, f'$h$', color='green', va='center')

    # Etiquetas de los terminales
    ax.text(-L/2 + s_w/2, Z/2, h + 1, "S", ha='center', fontweight='bold')
    ax.text(0, Z/2, h + 1, "G", ha='center', fontweight='bold')
    ax.text(0, -1, h - 0.9, "W(x)", ha='center', fontweight='bold')
    ax.text(L/2 - d_w/2, Z/2, h + 1, "D", ha='center', fontweight='bold')

    # Configuración de cámara y estética

    # Ajustes de cámara y ejes
    ax.set_xlim(-total_width/2 - 1, total_width/2 + 1)
    ax.set_ylim(-1, Z + 1)
  # Ponemos -1 arriba y h + substrate abajo.
    ax.set_zlim(-1, h + substrate_thickness)

    # Quitar los números (ticks) pero dejar las etiquetas de los ejes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.view_init(elev=25, azim=-45)
    plt.title("Representación 3D del Dispositivo")
    plt.show()

# Ejemplo de uso
draw_mesfet_total_width(L=12.0, Z=7.0, h=2.5)