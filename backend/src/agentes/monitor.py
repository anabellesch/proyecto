"""
Módulo del agente de monitoreo para la predicción de congestión de red.
Responsable de recolectar métricas de red entre dos equipos.
"""

import os
import time
import socket
import subprocess
import numpy as np
import logging
import json
from datetime import datetime
import psutil
import threading
import queue

# Importar configuración
from config.settings import MONITOREO, RED, UMBRAL_CONGESTION

# Configurar logger
logger = logging.getLogger(__name__)

class AgenteMonitoreo:
    """
    Agente que monitoriza los parámetros de red entre el equipo local
    y un equipo objetivo para detectar congestión.
    """
    
    def __init__(self, id_equipo, ip_objetivo=None):
        """
        Inicializa el agente de monitoreo.
        
        Args:
            id_equipo (str): Identificador único para este agente
            ip_objetivo (str, opcional): IP del equipo objetivo. Si es None,
                                        se usa el valor de la configuración.
        """
        self.id_equipo = id_equipo
        self.ip_objetivo = ip_objetivo or RED['IP_OBJETIVO']
        self.datos = []
        self.datos_queue = queue.Queue()
        self.ejecutando = False
        self.hilo_envio = None
        self.ancho_banda_maximo = None
        
        logger.info(f"Agente de monitoreo {id_equipo} inicializado. Objetivo: {self.ip_objetivo}")
    
    def inicializar(self):
        """Realizar inicialización completa del agente"""
        try:
            logger.info("Midiendo ancho de banda máximo...")
            self.ancho_banda_maximo = self._medir_ancho_banda_maximo()
            logger.info(f"Ancho de banda máximo: {self.ancho_banda_maximo} Mbps")
            return True
        except Exception as e:
            logger.error(f"Error durante la inicialización del agente: {e}")
            return False
    
    def _medir_ancho_banda_maximo(self):
        """
        Mide el ancho de banda máximo disponible usando iperf3
        
        Returns:
            float: Ancho de banda máximo en Mbps
        """
        try:
            comando = [
                MONITOREO['COMANDO_IPERF'], 
                "-c", self.ip_objetivo, 
                "-t", "5", 
                "-J"  # Formato JSON
            ]
            
            logger.debug(f"Ejecutando: {' '.join(comando)}")
            resultado = subprocess.run(
                comando,
                capture_output=True, text=True, check=True
            )
            
            # Parsear la salida JSON de iperf3
            datos = json.loads(resultado.stdout)
            # La estructura puede variar según versiones de iperf
            try:
                ancho_banda = datos['end']['sum_received']['bits_per_second'] / 1_000_000  # Convertir a Mbps
            except KeyError:
                logger.warning("Estructura JSON de iperf3 no reconocida, usando valor estimado")
                ancho_banda = 100  # Valor estimado
                
            return ancho_banda
            
        except (subprocess.SubprocessError, json.JSONDecodeError) as e:
            logger.warning(f"Error midiendo ancho de banda con iperf3: {e}")
            # Intentar conexión TCP básica como fallback
            try:
                inicio = time.time()
                with socket.create_connection((self.ip_objetivo, 80), timeout=2) as s:
                    pass
                tiempo_conexion = time.time() - inicio
                # Estimación muy aproximada basada en tiempo de conexión
                estimacion = 100 / (tiempo_conexion + 0.1)  # Evitar división por cero
                logger.info(f"Ancho de banda estimado por tiempo de conexión: {estimacion:.1f} Mbps")
                return min(estimacion, 1000)  # Limitar a un máximo razonable
            except (socket.error, socket.timeout):
                logger.error("No se pudo estimar el ancho de banda")
                return 100  # Valor predeterminado
    
    def medir_latencia(self):
        """
        Mide la latencia (RTT) hacia el equipo objetivo usando ping
        
        Returns:
            float: Latencia en milisegundos
        """
        try:
            # Adaptar comando ping según sistema operativo
            cmd = ["ping", "-c", str(MONITOREO['NUM_PINGS']), self.ip_objetivo]
            if os.name == 'nt':  # Windows
                cmd = ["ping", "-n", str(MONITOREO['NUM_PINGS']), self.ip_objetivo]
            
            logger.debug(f"Ejecutando comando ping: {' '.join(cmd)}")
            resultado = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extraer el tiempo promedio
            for linea in resultado.stdout.split('\n'):
                if 'avg' in linea or 'media' in linea or 'Average' in linea:
                    # Extraer el valor promedio de RTT
                    partes = linea.split('/')
                    if len(partes) >= 5:
                        return float(partes[4])
                    
            logger.warning("No se pudo extraer la latencia promedio de la salida de ping")
            return 0
        except Exception as e:
            logger.error(f"Error midiendo latencia: {e}")
            return 0
    
    def medir_perdida_paquetes(self):
        """
        Mide la pérdida de paquetes hacia el equipo objetivo
        
        Returns:
            float: Porcentaje de pérdida de paquetes
        """
        try:
            # Adaptar comando ping según sistema operativo
            cmd = ["ping", "-c", "10", self.ip_objetivo]
            if os.name == 'nt':  # Windows
                cmd = ["ping", "-n", "10", self.ip_objetivo]
            
            logger.debug(f"Ejecutando comando ping para pérdida: {' '.join(cmd)}")
            resultado = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extraer el porcentaje de pérdida
            for linea in resultado.stdout.split('\n'):
                if 'loss' in linea or 'perdidos' in linea:
                    # Extraer porcentaje
                    for parte in linea.split():
                        if '%' in parte:
                            return float(parte.strip('%'))
            
            logger.warning("No se pudo extraer el porcentaje de pérdida de paquetes")
            return 0
        except Exception as e:
            logger.error(f"Error midiendo pérdida de paquetes: {e}")
            return 0
    
    def medir_ancho_banda_actual(self):
        """
        Mide el ancho de banda actual usando iperf3 (versión rápida)
        
        Returns:
            float: Ancho de banda en Mbps
        """
        try:
            comando = [
                MONITOREO['COMANDO_IPERF'], 
                "-c", self.ip_objetivo, 
                "-t", "2",  # Test corto de 2 segundos
                "-J"
            ]
            
            logger.debug("Midiendo ancho de banda actual")
            resultado = subprocess.run(
                comando,
                capture_output=True, text=True, check=True
            )
            
            datos = json.loads(resultado.stdout)
            try:
                ancho_banda = datos['end']['sum_received']['bits_per_second'] / 1_000_000  # Convertir a Mbps
                return ancho_banda
            except KeyError:
                logger.warning("Estructura JSON de iperf3 no reconocida")
                return self.ancho_banda_maximo * 0.8  # Estimación
                
        except Exception as e:
            logger.warning(f"Error midiendo ancho de banda actual: {e}")
            # En caso de error, hacer una estimación basada en el máximo
            if self.ancho_banda_maximo:
                return self.ancho_banda_maximo * (0.5 + 0.5 * np.random.random())
            else:
                return 50  # Valor por defecto
    
    def medir_jitter(self):
        """
        Mide la variación en el retardo (jitter)
        
        Returns:
            float: Jitter en milisegundos
        """
        try:
            cmd = ["ping", "-c", "10", self.ip_objetivo]
            if os.name == 'nt':  # Windows
                # Windows no tiene una manera directa de obtener jitter con ping
                # Se podría implementar manualmente recolectando tiempos y calculando la desviación
                return np.random.random() * 5  # Simulación para Windows
            
            resultado = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extraer tiempos individuales y calcular jitter
            tiempos = []
            for linea in resultado.stdout.split('\n'):
                if 'time=' in linea or 'tiempo=' in linea:
                    for parte in linea.split():
                        if 'time=' in parte or 'tiempo=' in parte:
                            tiempo = float(parte.split('=')[1].replace('ms', ''))
                            tiempos.append(tiempo)
            
            if len(tiempos) >= 2:
                # Calcular variaciones entre tiempos consecutivos
                diferencias = [abs(tiempos[i] - tiempos[i-1]) for i in range(1, len(tiempos))]
                return np.mean(diferencias)  # Promedio de variaciones
            else:
                logger.warning("No se pudieron obtener suficientes muestras para calcular jitter")
                return 0
        except Exception as e:
            logger.error(f"Error midiendo jitter: {e}")
            return 0
    
    def obtener_metricas_sistema(self):
        """
        Obtiene métricas del sistema local
        
        Returns:
            dict: Diccionario con métricas del sistema
        """
        try:
            # Recopilar información del sistema
            cpu = psutil.cpu_percent()
            memoria = psutil.virtual_memory().percent
            
            # Intentar obtener métricas de disco
            try:
                io_disk = psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes
            except Exception:
                io_disk = 0
                
            # Contar conexiones de red activas
            try:
                conexiones = len([c for c in psutil.net_connections() if c.status == 'ESTABLISHED'])
            except Exception:
                conexiones = 0
            
            return {
                'cpu': cpu,
                'memoria': memoria,
                'conexiones': conexiones,
                'io_disk': io_disk
            }
        except Exception as e:
            logger.error(f"Error obteniendo métricas del sistema: {e}")
            return {
                'cpu': 0,
                'memoria': 0,
                'conexiones': 0,
                'io_disk': 0
            }
    
    def recolectar_datos(self):
        """
        Recolecta todos los datos de red y sistema
        
        Returns:
            dict: Diccionario con todas las métricas
        """
        timestamp = datetime.now()
        
        # Recopilar métricas
        latencia = self.medir_latencia()
        perdida = self.medir_perdida_paquetes()
        
        # Mediciones más intensivas solo cada cierto número de iteraciones
        # para no sobrecargar la red
        if len(self.datos) % 5 == 0:  # Cada 5 ciclos
            ancho_banda = self.medir_ancho_banda_actual()
            jitter = self.medir_jitter()
        elif self.datos:  # Si hay datos previos, usar el último valor
            ancho_banda = self.datos[-1].get('ancho_banda', 0)
            jitter = self.datos[-1].get('jitter', 0)
        else:  # Primera ejecución
            ancho_banda = 0
            jitter = 0
            
        sistema = self.obtener_metricas_sistema()
        
        # Crear diccionario de datos
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
        
        # Registrar métricas principales
        logger.debug(f"Métricas: Latencia={latencia:.1f}ms, "
                    f"Pérdida={perdida:.1f}%, "
                    f"Ancho de banda={ancho_banda:.1f}Mbps")
        
        # Añadir a la lista de datos y a la cola de envío
        self.datos.append(datos)
        # Limitar el tamaño de datos en memoria
        if len(self.datos) > MONITOREO['MAX_MUESTRAS_MEMORIA']:
            self.datos.pop(0)
            
        self.datos_queue.put(datos)
        
        return datos
    
    def _hilo_envio_datos(self, servidor, puerto):
        """
        Hilo para enviar datos al servidor central
        
        Args:
            servidor (str): Dirección IP del servidor central
            puerto (int): Puerto del servidor central
        """
        logger.info(f"Iniciando hilo de envío de datos a {servidor}:{puerto}")
        
        while self.ejecutando:
            try:
                # Esperar por nuevos datos (timeout para poder detener el hilo)
                try:
                    datos = self.datos_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Enviar datos al servidor
                self._enviar_datos_a_servidor(datos, servidor, puerto)
                
            except Exception as e:
                logger.error(f"Error en hilo de envío: {e}")
                time.sleep(5)  # Esperar antes de reintentar
    
    def _enviar_datos_a_servidor(self, datos, servidor, puerto):
        """
        Envía datos al servidor central
        
        Args:
            datos (dict): Datos a enviar
            servidor (str): Dirección IP del servidor
            puerto (int): Puerto del servidor
        """
        try:
            # Preparar datos para envío
            datos_envio = datos.copy()
            datos_envio['timestamp'] = datos_envio['timestamp'].isoformat()
            
            # Conectar al servidor
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)
                s.connect((servidor, puerto))
                
                # Enviar datos
                mensaje = json.dumps(datos_envio).encode('utf-8')
                s.sendall(mensaje)
                
                # Recibir confirmación
                respuesta = s.recv(1024).decode('utf-8')
                if respuesta != "OK":
                    logger.warning(f"Respuesta inesperada del servidor: {respuesta}")
                    
            logger.debug(f"Datos enviados a {servidor}:{puerto}")
            
        except (socket.timeout, ConnectionRefusedError) as e:
            logger.error(f"Error de conexión con el servidor: {e}")
        except Exception as e:
            logger.error(f"Error enviando datos: {e}")
    
    def iniciar_monitoreo(self, servidor=None, puerto=None, duracion=None):
        """
        Inicia el monitoreo continuo
        
        Args:
            servidor (str, opcional): Servidor de datos. Si es None, usa configuración.
            puerto (int, opcional): Puerto del servidor. Si es None, usa configuración.
            duracion (float, opcional): Duración en segundos. Si es None, continúa indefinidamente.
            
        Returns:
            list: Lista de datos recolectados (si se especificó duración)
        """
        if not self.ancho_banda_maximo:
            if not self.inicializar():
                logger.error("No se pudo inicializar el agente de monitoreo")
                return []
        
        # Usar valores de configuración si no se especifican
        servidor = servidor or RED['SERVIDOR_DATOS']
        puerto = puerto or RED['PUERTO_DATOS']
        
        self.ejecutando = True
        
        # Iniciar hilo de envío si hay un servidor configurado
        if servidor and puerto:
            self.hilo_envio = threading.Thread(
                target=self._hilo_envio_datos,
                args=(servidor, puerto),
                daemon=True
            )
            self.hilo_envio.start()
        
        logger.info(f"Iniciando monitoreo{' por ' + str(duracion) + ' segundos' if duracion else ''}")
        inicio = time.time()
        
        try:
            while self.ejecutando and (duracion is None or (time.time() - inicio) < duracion):
                # Recolectar datos
                self.recolectar_datos()
                
                # Esperar hasta el siguiente intervalo de muestreo
                time.sleep(MONITOREO['INTERVALO_MUESTREO'])
        
        except KeyboardInterrupt:
            logger.info("Monitoreo interrumpido por el usuario")
        finally:
            self.detener_monitoreo()
            
        return self.datos
    
    def detener_monitoreo(self):
        """Detiene el monitoreo y los hilos asociados"""
        logger.info("Deteniendo monitore")