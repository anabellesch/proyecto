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