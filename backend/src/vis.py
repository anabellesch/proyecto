import os
import time
import socket
import subprocess
import numpy as np
import pandas as pd
import psutil
import threading
import pickle
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ================================
# VISUALIZACIÓN
# ================================

def visualizar_metricas(datos):
    """Visualiza las métricas principales de red"""
    df = pd.DataFrame(datos)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Crear figura
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Gráfico 1: Latencia
    axs[0].plot(df['timestamp'], df['latencia'], 'b-', label='Latencia (ms)')
    axs[0].axhline(y=UMBRAL_CONGESTION['latencia'], color='r', linestyle='--', 
                  label=f'Umbral ({UMBRAL_CONGESTION["latencia"]} ms)')
    axs[0].set_ylabel('Latencia (ms)')
    axs[0].set_title('Latencia de Red')
    axs[0].legend()
    axs[0].grid(True)
    
    # Gráfico 2: Pérdida de paquetes
    axs[1].plot(df['timestamp'], df['perdida_paquetes'], 'g-', label='Pérdida de paquetes (%)')
    axs[1].axhline(y=UMBRAL_CONGESTION['perdida_paquetes'], color='r', linestyle='--',
                  label=f'Umbral ({UMBRAL_CONGESTION["perdida_paquetes"]}%)')
    axs[1].set_ylabel('Pérdida (%)')
    axs[1].set_title('Pérdida de Paquetes')
    axs[1].legend()
    axs[1].grid(True)
    
    # Gráfico 3: Ancho de banda
    axs[2].plot(df['timestamp'], df['ancho_banda'], 'm-', label='Ancho de banda (Mbps)')
    # El umbral es un porcentaje del máximo, por lo que calculamos el valor absoluto
    umbral_abs = df['ancho_banda'].max() * UMBRAL_CONGESTION['ancho_banda'] / 100
    axs[2].axhline(y=umbral_abs, color='r', linestyle='--',
                  label=f'Umbral ({UMBRAL_CONGESTION["ancho_banda"]}% del máx)')
    axs[2].set_ylabel('Ancho de banda (Mbps)')
    axs[2].set_xlabel('Tiempo')
    axs[2].set_title('Ancho de Banda')
    axs[2].legend()
    axs[2].grid(True)
    
    # Ajustar espaciado
    plt.tight_layout()
    plt.show()

# ================================
# SIMULACIÓN DEL SISTEMA COMPLETO
# ================================

def simular_congestion(duracion_mins=5):
    """Simula un escenario completo incluyendo entrenamiento y predicción"""
    print(f"Iniciando simulación de {duracion_mins} minutos...")
    
    # Crear y configurar componentes
    agente = AgenteMonitoreo(id_equipo="PC-01", ip_objetivo="192.168.1.100")
    procesador = ProcesadorDatos()
    modelo = ModeloPrediccion()
    alertas = SistemaAlertas(umbral_alerta=0.7)
    
    # Recolectar datos iniciales (fase de entrenamiento)
    print("Recolectando datos para entrenamiento...")
    datos_entrenamiento = agente.iniciar_monitoreo(duracion=60)  # 1 minuto
    
    # Procesar datos para entrenamiento
    df = procesador.preparar_datos(datos_entrenamiento)
    df = procesador.crear_etiquetas_congestion(df)
    
    # Crear ventanas de tiempo para entrenamiento
    X, y = procesador.crear_ventanas_temporales(df, ventana=10)
    
    if len(X) > 0:
        # Entrenar modelo
        print("Entrenando modelo de predicción...")
        modelo.entrenar(X, y)
        
        # Iniciar monitoreo en tiempo real con predicciones
        print("Iniciando monitoreo y predicción en tiempo real...")
        
        for _ in range(duracion_mins * 60 // INTERVALO_MUESTREO):
            # Recolectar nuevos datos
            nuevo_dato = agente.recolectar_datos()
            
            # Preparar para predicción
            df_reciente = procesador.preparar_datos(agente.datos[-15:])  # Últimos 15 puntos
            X_reciente, _ = procesador.crear_ventanas_temporales(df_reciente, ventana=10, horizonte=1)
            
            if len(X_reciente) > 0:
                # Predecir congestión
                _, prob = modelo.predecir(X_reciente[-1:])
                
                # Evaluar necesidad de alertas
                alertas.evaluar_prediccion(prob[0])
                
                # Simulación: mostrar datos actuales
                print(f"Tiempo: {nuevo_dato['timestamp'].strftime('%H:%M:%S')} | "
                      f"Latencia: {nuevo_dato['latencia']:.1f}ms | "
                      f"Prob. Congestión: {prob[0]:.2%}")
            
            time.sleep(INTERVALO_MUESTREO)
        
        # Visualizar datos al final
        visualizar_metricas(agente.datos)
    else:
        print("No hay suficientes datos para entrenar el modelo")

# Ejecutar simulación si se ejecuta directamente
if __name__ == "__main__":
    simular_congestion(duracion_mins=2)