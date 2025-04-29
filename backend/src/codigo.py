"""
Proyecto de IA para Predicción de Congestión de Red
===================================================
Este código proporciona un ejemplo de implementación para el proyecto.
Incluye:
1. Agente de monitoreo de red
2. Procesamiento de datos
3. Modelo de predicción
4. Visualización básica
"""

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

# Configuración
SERVIDOR_DATOS = "192.168.1.100"  # IP de la computadora que almacena datos
PUERTO_DATOS = 5555
INTERVALO_MUESTREO = 1  # segundos
VENTANA_PREDICCION = 60  # predecir congestión en los próximos 60 segundos
UMBRAL_CONGESTION = {
    'latencia': 100,  # ms
    'perdida_paquetes': 2.0,  # %
    'ancho_banda': 50  # % del máximo
}

# ================================
# AGENTE DE MONITOREO
# ================================

class AgenteMonitoreo:
    def __init__(self, id_equipo, ip_objetivo):
        self.id_equipo = id_equipo
        self.ip_objetivo = ip_objetivo
        self.ancho_banda_maximo = self._medir_ancho_banda_maximo()
        self.datos = []
        
    def _medir_ancho_banda_maximo(self):
        """Mide el ancho de banda máximo disponible usando iperf3"""
        try:
            # Esto asume que hay un servidor iperf3 en el equipo objetivo
            resultado = subprocess.run(
                ["iperf3", "-c", self.ip_objetivo, "-t", "5", "-J"],
                capture_output=True, text=True, check=True
            )
            # En una implementación real, parseamos la salida JSON
            return 100  # Mbps (valor simulado)
        except Exception as e:
            print(f"Error midiendo ancho de banda: {e}")
            return 100  # Valor predeterminado
    
    def medir_latencia(self):
        """Mide la latencia (RTT) hacia el equipo objetivo usando ping"""
        try:
            cmd = ["ping", "-c", "5", self.ip_objetivo]
            if os.name == 'nt':  # Windows
                cmd = ["ping", "-n", "5", self.ip_objetivo]
            
            resultado = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extraer el tiempo promedio (simplificado)
            for linea in resultado.stdout.split('\n'):
                if 'avg' in linea or 'media' in linea or 'Average' in linea:
                    # Extraer el valor promedio de RTT
                    partes = linea.split('/')
                    if len(partes) >= 5:
                        return float(partes[4])
            return 0
        except Exception as e:
            print(f"Error midiendo latencia: {e}")
            return 0
    
    def medir_perdida_paquetes(self):
        """Mide la pérdida de paquetes hacia el equipo objetivo"""
        try:
            cmd = ["ping", "-c", "10", self.ip_objetivo]
            if os.name == 'nt':  # Windows
                cmd = ["ping", "-n", "10", self.ip_objetivo]
            
            resultado = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extraer el porcentaje de pérdida (simplificado)
            for linea in resultado.stdout.split('\n'):
                if 'loss' in linea or 'perdidos' in linea:
                    # Extraer porcentaje
                    for parte in linea.split():
                        if '%' in parte:
                            return float(parte.strip('%'))
            return 0
        except Exception as e:
            print(f"Error midiendo pérdida de paquetes: {e}")
            return 0
    
    def medir_ancho_banda_actual(self):
        """Mide el ancho de banda actual usando iperf3"""
        try:
            # Simulación simplificada
            return self.ancho_banda_maximo * (0.5 + 0.5 * np.random.random())
        except Exception as e:
            print(f"Error midiendo ancho de banda actual: {e}")
            return 0
    
    def medir_jitter(self):
        """Mide la variación en el retardo (jitter)"""
        # Simulación simplificada
        return np.random.random() * 10  # ms
    
    def obtener_metricas_sistema(self):
        """Obtiene métricas del sistema local"""
        return {
            'cpu': psutil.cpu_percent(),
            'memoria': psutil.virtual_memory().percent,
            'conexiones': len(psutil.net_connections()),
            'io_disk': psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes
        }
    
    def recolectar_datos(self):
        """Recolecta todos los datos de red y sistema"""
        timestamp = datetime.now()
        latencia = self.medir_latencia()
        perdida = self.medir_perdida_paquetes()
        ancho_banda = self.medir_ancho_banda_actual()
        jitter = self.medir_jitter()
        sistema = self.obtener_metricas_sistema()
        
        datos = {
            'timestamp': timestamp,
            'id_equipo': self.id_equipo,
            'latencia': latencia,
            'perdida_paquetes': perdida,
            'ancho_banda': ancho_banda,
            'jitter': jitter,
            'cpu': sistema['cpu'],
            'memoria': sistema['memoria'],
            'conexiones': sistema['conexiones'],
            'io_disk': sistema['io_disk']
        }
        
        self.datos.append(datos)
        return datos
    
    def enviar_datos(self, datos):
        """Envía los datos recolectados al servidor central"""
        try:
            # Simulación simplificada
            print(f"Enviando datos: {datos['timestamp']} - Latencia: {datos['latencia']}ms")
            # En una implementación real, usaríamos sockets o API REST
        except Exception as e:
            print(f"Error enviando datos: {e}")
    
    def iniciar_monitoreo(self, duracion=None):
        """Inicia el monitoreo continuo"""
        inicio = time.time()
        
        while duracion is None or (time.time() - inicio) < duracion:
            datos = self.recolectar_datos()
            self.enviar_datos(datos)
            time.sleep(INTERVALO_MUESTREO)
            
        return self.datos

# ================================
# PROCESAMIENTO DE DATOS
# ================================

class ProcesadorDatos:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preparar_datos(self, datos_crudos):
        """Convierte los datos crudos a DataFrame y realiza preprocesamiento"""
        # Convertir a DataFrame
        df = pd.DataFrame(datos_crudos)
        
        # Asegurar que timestamp sea índice datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Eliminar columnas no numéricas (excepto timestamp que ya es índice)
        df = df.select_dtypes(include=['number'])
        
        # Detectar y manejar valores faltantes
        df.fillna(method='ffill', inplace=True)  # Forward fill
        df.fillna(df.mean(), inplace=True)  # Rellenar con promedio si aún hay NaN
        
        # Agregar características derivadas
        if len(df) > 1:
            # Cambios porcentuales
            for col in ['latencia', 'ancho_banda', 'perdida_paquetes']:
                if col in df.columns:
                    df[f'{col}_pct_cambio'] = df[col].pct_change().fillna(0)
            
            # Medias móviles
            for col in df.columns:
                if df[col].dtype in [np.float64, np.int64]:
                    df[f'{col}_media_5'] = df[col].rolling(window=5, min_periods=1).mean()
        
        return df
    
    def normalizar_datos(self, df):
        """Normaliza los datos usando StandardScaler"""
        # Guardar nombres de columnas
        columnas = df.columns
        
        # Aplicar scaling
        datos_norm = self.scaler.fit_transform(df)
        
        # Devolver como DataFrame
        return pd.DataFrame(datos_norm, index=df.index, columns=columnas)
    
    def crear_etiquetas_congestion(self, df):
        """Crear etiquetas binarias para situaciones de congestión"""
        # Definir condiciones de congestión basadas en umbrales
        condiciones = []
        
        if 'latencia' in df.columns:
            condiciones.append(df['latencia'] > UMBRAL_CONGESTION['latencia'])
        
        if 'perdida_paquetes' in df.columns:
            condiciones.append(df['perdida_paquetes'] > UMBRAL_CONGESTION['perdida_paquetes'])
        
        if 'ancho_banda' in df.columns:
            # Convertir a porcentaje del máximo
            banda_rel = df['ancho_banda'] / df['ancho_banda'].max() * 100
            condiciones.append(banda_rel < UMBRAL_CONGESTION['ancho_banda'])
        
        # Combinación de condiciones (al menos una condición se cumple)
        if condiciones:
            df['congestion'] = np.logical_or.reduce(condiciones).astype(int)
        else:
            df['congestion'] = 0
        
        return df
    
    def crear_ventanas_temporales(self, df, ventana=10, horizonte=VENTANA_PREDICCION):
        """
        Crea ventanas temporales para entrenamiento:
        - Cada fila X contiene 'ventana' intervalos de tiempo de datos
        - Cada etiqueta y indica si hay congestión en los próximos 'horizonte' intervalos
        """
        X = []
        y = []
        
        if 'congestion' not in df.columns:
            df = self.crear_etiquetas_congestion(df)
        
        # Columnas a usar como características (excluir la etiqueta)
        columnas = [col for col in df.columns if col != 'congestion']
        
        for i in range(len(df) - ventana - horizonte + 1):
            # Ventana de características
            ventana_datos = df.iloc[i:i+ventana][columnas].values.flatten()
            X.append(ventana_datos)
            
            # Etiqueta: ¿Hay congestión en el horizonte de predicción?
            congestion_futura = df.iloc[i+ventana:i+ventana+horizonte]['congestion'].max()
            y.append(congestion_futura)
        
        return np.array(X), np.array(y)

# ================================
# MODELO DE PREDICCIÓN
# ================================

class ModeloPrediccion:
    def __init__(self):
        # Modelo base: Random Forest para clasificación
        self.modelo = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.umbral = 0.5  # Umbral para clasificación binaria
        self.entrenado = False
    
    def entrenar(self, X, y):
        """Entrena el modelo con los datos proporcionados"""
        # Dividir en conjuntos de entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entrenar modelo
        self.modelo.fit(X_train, y_train)
        
        # Evaluar en validación
        preds = self.modelo.predict(X_val)
        preds_bin = (preds >= self.umbral).astype(int)
        
        # Métricas básicas
        precision = np.mean(preds_bin == y_val)
        print(f"Precisión en validación: {precision:.4f}")
        
        self.entrenado = True
        return precision
    
    def predecir(self, X):
        """Realiza predicciones con el modelo entrenado"""
        if not self.entrenado:
            raise ValueError("El modelo no ha sido entrenado")
        
        # Obtener probabilidades de congestión
        probs = self.modelo.predict(X)
        
        # Convertir a predicciones binarias según umbral
        preds = (probs >= self.umbral).astype(int)
        
        return preds, probs
    
    def guardar_modelo(self, ruta):
        """Guarda el modelo entrenado en disco"""
        if not self.entrenado:
            raise ValueError("No se puede guardar un modelo no entrenado")
        
        with open(ruta, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def cargar_modelo(cls, ruta):
        """Carga un modelo entrenado desde disco"""
        with open(ruta, 'rb') as f:
            modelo = pickle.load(f)
        
        return modelo

# ================================
# SISTEMA DE ALERTAS
# ================================

class SistemaAlertas:
    def __init__(self, umbral_alerta=0.7):
        self.umbral_alerta = umbral_alerta
        self.alertas_activas = False
    
    def evaluar_prediccion(self, prob_congestion):
        """Evalúa si la predicción amerita generar una alerta"""
        if prob_congestion >= self.umbral_alerta and not self.alertas_activas:
            self._generar_alerta(prob_congestion)
            self.alertas_activas = True
        elif prob_congestion < self.umbral_alerta and self.alertas_activas:
            self._cancelar_alerta()
            self.alertas_activas = False
    
    def _generar_alerta(self, probabilidad):
        """Genera una alerta de congestión inminente"""
        mensaje = f"¡ALERTA! Congestión de red probable ({probabilidad:.2%})"
        print(f"\n{'!'*50}\n{mensaje}\n{'!'*50}\n")
        
        # Aquí se implementarían otros métodos de notificación:
        # - Correo electrónico
        # - SMS
        # - Integración con sistemas de monitoreo
        # - Etc.
    
    def _cancelar_alerta(self):
        """Cancela la alerta activa"""
        print("\n--- Alerta de congestión cancelada ---\n")

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