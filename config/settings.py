import os
from dotenv import load_dotenv
import logging

# Cargar variables de entorno desde archivo .env si existe
load_dotenv()

# Configuración básica
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Asegurar que existen los directorios necesarios
for directory in [DATA_DIR, LOGS_DIR, os.path.join(DATA_DIR, 'models'), 
                  os.path.join(DATA_DIR, 'collected')]:
    os.makedirs(directory, exist_ok=True)

# Configuración de red
RED = {
    # IP del equipo remoto para monitorizar (obtener de variable de entorno o usar valor por defecto)
    'IP_OBJETIVO': os.getenv('RED_IP_OBJETIVO', '192.168.1.100'),
    
    # IP del servidor central de datos
    'SERVIDOR_DATOS': os.getenv('RED_SERVIDOR_DATOS', '192.168.1.100'),
    
    # Puerto para comunicación con el servidor de datos
    'PUERTO_DATOS': int(os.getenv('RED_PUERTO_DATOS', '5555')),
    
    # Puerto para servidor web de visualización
    'PUERTO_WEB': int(os.getenv('RED_PUERTO_WEB', '8080')),
}

# Configuración del monitoreo
MONITOREO = {
    # Intervalo entre mediciones (en segundos)
    'INTERVALO_MUESTREO': float(os.getenv('MONITOR_INTERVALO', '1')),
    
    # Número de muestras a mantener en memoria
    'MAX_MUESTRAS_MEMORIA': int(os.getenv('MONITOR_MAX_MUESTRAS', '3600')),
    
    # Número de pings para cada medición de latencia
    'NUM_PINGS': int(os.getenv('MONITOR_NUM_PINGS', '5')),
    
    # Comandos para mediciones específicas de la plataforma
    'COMANDO_IPERF': os.getenv('MONITOR_COMANDO_IPERF', 'iperf3'),
}

# Configuración del modelo de predicción
MODELO = {
    # Ventana temporal para predicción (en segundos)
    'VENTANA_PREDICCION': int(os.getenv('MODELO_VENTANA', '60')),
    
    # Tamaño de la ventana de datos para entrenamiento
    'VENTANA_ENTRENAMIENTO': int(os.getenv('MODELO_VENTANA_ENTRENAMIENTO', '10')),
    
    # Ruta al modelo guardado (si existe)
    'RUTA_MODELO': os.path.join(DATA_DIR, 'models', 'modelo_congestion.pkl'),
    
    # Tipo de algoritmo a utilizar ('random_forest', 'lstm', etc.)
    'TIPO_ALGORITMO': os.getenv('MODELO_ALGORITMO', 'random_forest'),
    
    # Parámetros para Random Forest
    'RF_N_ESTIMADORES': int(os.getenv('MODELO_RF_ESTIMADORES', '100')),
    'RF_MAX_DEPTH': int(os.getenv('MODELO_RF_MAX_DEPTH', '10')),
    
    # Frecuencia de reentrenamiento (en número de nuevas muestras)
    'FRECUENCIA_REENTRENAMIENTO': int(os.getenv('MODELO_REENTRENAMIENTO', '1000')),
}

# Umbrales para determinar congestión
UMBRAL_CONGESTION = {
    'latencia': float(os.getenv('UMBRAL_LATENCIA', '100')),  # ms
    'perdida_paquetes': float(os.getenv('UMBRAL_PERDIDA', '2.0')),  # %
    'ancho_banda': float(os.getenv('UMBRAL_ANCHO_BANDA', '50')),  # % del máximo
    'jitter': float(os.getenv('UMBRAL_JITTER', '20')),  # ms
}

# Configuración de alertas
ALERTAS = {
    # Umbral de probabilidad para generar alerta
    'UMBRAL_PROB_ALERTA': float(os.getenv('ALERTAS_UMBRAL', '0.7')),
    
    # Tiempo mínimo entre alertas consecutivas (en segundos)
    'TIEMPO_MINIMO_ENTRE_ALERTAS': int(os.getenv('ALERTAS_TIEMPO_MIN', '300')),
    
    # Habilitar/deshabilitar tipos de notificaciones
    'NOTIFICACION_CONSOLA': os.getenv('ALERTAS_CONSOLA', 'true').lower() == 'true',
    'NOTIFICACION_EMAIL': os.getenv('ALERTAS_EMAIL', 'false').lower() == 'true',
    'NOTIFICACION_SMS': os.getenv('ALERTAS_SMS', 'false').lower() == 'true',
    
    # Configuración de correo (si está habilitado)
    'EMAIL_DESTINATARIOS': os.getenv('ALERTAS_EMAIL_DESTINATARIOS', '').split(','),
    'EMAIL_REMITENTE': os.getenv('ALERTAS_EMAIL_REMITENTE', 'alertas@example.com'),
    'EMAIL_SERVIDOR': os.getenv('ALERTAS_EMAIL_SERVIDOR', 'smtp.example.com'),
    'EMAIL_PUERTO': int(os.getenv('ALERTAS_EMAIL_PUERTO', '587')),
}

# Configuración de logging
LOGGING = {
    'NIVEL': os.getenv('LOG_NIVEL', 'INFO'),
    'FORMATO': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'ARCHIVO': os.path.join(LOGS_DIR, 'prediccion_congestion.log'),
    'MAX_TAMANO': 10 * 1024 * 1024,  # 10 MB
    'NUM_BACKUPS': 5,
}

# Configuración específica para desarrollo/producción
ENTORNO = os.getenv('ENTORNO', 'desarrollo')
DEBUG = ENTORNO.lower() == 'desarrollo'

# Configuración específica según el entorno
if DEBUG:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=getattr(logging, LOGGING['NIVEL']))