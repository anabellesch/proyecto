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
