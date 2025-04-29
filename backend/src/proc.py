
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
from src import agente



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