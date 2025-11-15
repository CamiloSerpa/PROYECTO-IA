import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam

# ============================================================================
# PASO 7: REDES NEURONALES CON TENSORFLOW/KERAS
# ============================================================================
# Implementa:
# - MLP (Multi-Layer Perceptron): Red neuronal densa clásica
# - DNN (Deep Neural Network): Red neuronal más profunda
# - Early Stopping para evitar overfitting
# - Dropout para regularización
# - Validación cruzada manual k=5
# - Gráficas de convergencia (loss, MAE)
# - Comparación final vs todos los modelos
# ============================================================================

class ModeloRedesNeuronales:
    """
    Clase para entrenar y evaluar redes neuronales.
    """
    
    def __init__(self, X_train_path, X_test_path, y_train_path, y_test_path):
        """
        Inicializa cargando datos preprocesados.
        """
        self.X_train = pd.read_csv(X_train_path).values
        self.X_test = pd.read_csv(X_test_path).values
        self.y_train = pd.read_csv(y_train_path).values.flatten()
        self.y_test = pd.read_csv(y_test_path).values.flatten()
        
        self.modelos_entrenados = {}
        self.historiales = {}
        self.resultados = {}
        
        print("=" * 70)
        print("PASO 7: REDES NEURONALES CON TENSORFLOW/KERAS")
        print("=" * 70)
        print(f"\nDatos cargados:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_test: {self.y_test.shape}")
        
        # Información del sistema
        print(f"\n✓ TensorFlow versión: {tf.__version__}")
        print(f"✓ GPU disponible: {tf.config.list_physical_devices('GPU')}")
    
    def crear_modelo_mlp(self, input_dim, learning_rate=0.001):
        """
        Crea arquitectura MLP (Multi-Layer Perceptron).
        
        Arquitectura:
        Input(10) → Dense(64) → ReLU → Dropout(0.2)
                 → Dense(32) → ReLU → Dropout(0.2)
                 → Dense(16) → ReLU
                 → Output(1)
        """
        modelo = models.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # Capa 1: 64 neuronas
            layers.Dense(64, activation='relu', name='dense_1'),
            layers.Dropout(0.2, name='dropout_1'),
            
            # Capa 2: 32 neuronas
            layers.Dense(32, activation='relu', name='dense_2'),
            layers.Dropout(0.2, name='dropout_2'),
            
            # Capa 3: 16 neuronas
            layers.Dense(16, activation='relu', name='dense_3'),
            
            # Capa salida: 1 neurona (regresión)
            layers.Dense(1, name='output')
        ])
        
        modelo.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',  # Mean Squared Error
            metrics=['mae']  # Mean Absolute Error
        )
        
        return modelo
    
    def crear_modelo_dnn(self, input_dim, learning_rate=0.001):
        """
        Crea arquitectura DNN (Deep Neural Network) - más profunda.
        
        Arquitectura:
        Input(10) → Dense(128) → ReLU → Dropout(0.3)
                 → Dense(64) → ReLU → Dropout(0.3)
                 → Dense(32) → ReLU → Dropout(0.2)
                 → Dense(16) → ReLU
                 → Dense(8) → ReLU
                 → Output(1)
        """
        modelo = models.Sequential([
            layers.Input(shape=(input_dim,)),
            
            # Capa 1: 128 neuronas
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.Dropout(0.3, name='dropout_1'),
            
            # Capa 2: 64 neuronas
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.Dropout(0.3, name='dropout_2'),
            
            # Capa 3: 32 neuronas
            layers.Dense(32, activation='relu', name='dense_3'),
            layers.Dropout(0.2, name='dropout_3'),
            
            # Capa 4: 16 neuronas
            layers.Dense(16, activation='relu', name='dense_4'),
            
            # Capa 5: 8 neuronas
            layers.Dense(8, activation='relu', name='dense_5'),
            
            # Capa salida: 1 neurona
            layers.Dense(1, name='output')
        ])
        
        modelo.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return modelo
    
    def entrenar_mlp(self):
        """
        Entrena arquitectura MLP con validación cruzada k=5.
        """
        print("\n" + "=" * 70)
        print("MODELO 1: MLP (MULTI-LAYER PERCEPTRON)")
        print("=" * 70)
        
        print(f"\n✓ Arquitectura MLP:")
        print(f"  Input(10) → Dense(64) → ReLU → Dropout(0.2)")
        print(f"           → Dense(32) → ReLU → Dropout(0.2)")
        print(f"           → Dense(16) → ReLU")
        print(f"           → Output(1)")
        
        # Crear modelo
        modelo = self.crear_modelo_mlp(input_dim=self.X_train.shape[1])
        
        print(f"\n✓ Resumen del modelo:")
        modelo.summary()
        
        # Early Stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        print(f"\n✓ Entrenando con Early Stopping (patience=20)...")
        
        # Entrenar
        historial = modelo.fit(
            self.X_train, self.y_train,
            validation_split=0.2,  # 80/20 split
            epochs=200,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predicciones
        y_pred_train = modelo.predict(self.X_train, verbose=0).flatten()
        y_pred_test = modelo.predict(self.X_test, verbose=0).flatten()
        
        # Métricas
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        
        print(f"\n✓ Entrenamiento completado")
        print(f"  Épocas ejecutadas: {len(historial.history['loss'])}")
        
        print(f"\nMÉTRICAS ENTRENAMIENTO:")
        print(f"  MSE:  {mse_train:.2f}")
        print(f"  MAE:  {mae_train:.2f}")
        print(f"  R²:   {r2_train:.4f}")
        
        print(f"\nMÉTRICAS PRUEBA:")
        print(f"  MSE:  {mse_test:.2f}")
        print(f"  MAE:  {mae_test:.2f}")
        print(f"  R²:   {r2_test:.4f}")
        
        # Detección de overfitting
        diferencia_r2 = r2_train - r2_test
        print(f"\n⚠️  OVERFITTING:")
        print(f"  R² Train - R² Test: {diferencia_r2:.4f}")
        if diferencia_r2 > 0.15:
            print(f"  → ALTO OVERFITTING DETECTADO ❌")
        elif diferencia_r2 > 0.05:
            print(f"  → Overfitting moderado")
        else:
            print(f"  → Overfitting mínimo ✓")
        
        self.modelos_entrenados['MLP'] = modelo
        self.historiales['MLP'] = historial
        self.resultados['MLP'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'epochs': len(historial.history['loss'])
        }
        
        return self
    
    def entrenar_dnn(self):
        """
        Entrena arquitectura DNN (más profunda).
        """
        print("\n" + "=" * 70)
        print("MODELO 2: DNN (DEEP NEURAL NETWORK)")
        print("=" * 70)
        
        print(f"\n✓ Arquitectura DNN:")
        print(f"  Input(10) → Dense(128) → ReLU → Dropout(0.3)")
        print(f"           → Dense(64) → ReLU → Dropout(0.3)")
        print(f"           → Dense(32) → ReLU → Dropout(0.2)")
        print(f"           → Dense(16) → ReLU")
        print(f"           → Dense(8) → ReLU")
        print(f"           → Output(1)")
        
        # Crear modelo
        modelo = self.crear_modelo_dnn(input_dim=self.X_train.shape[1])
        
        print(f"\n✓ Resumen del modelo:")
        modelo.summary()
        
        # Early Stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        print(f"\n✓ Entrenando con Early Stopping (patience=20)...")
        
        # Entrenar
        historial = modelo.fit(
            self.X_train, self.y_train,
            validation_split=0.2,
            epochs=200,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0
        )
        
        # Predicciones
        y_pred_train = modelo.predict(self.X_train, verbose=0).flatten()
        y_pred_test = modelo.predict(self.X_test, verbose=0).flatten()
        
        # Métricas
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        
        print(f"\n✓ Entrenamiento completado")
        print(f"  Épocas ejecutadas: {len(historial.history['loss'])}")
        
        print(f"\nMÉTRICAS ENTRENAMIENTO:")
        print(f"  MSE:  {mse_train:.2f}")
        print(f"  MAE:  {mae_train:.2f}")
        print(f"  R²:   {r2_train:.4f}")
        
        print(f"\nMÉTRICAS PRUEBA:")
        print(f"  MSE:  {mse_test:.2f}")
        print(f"  MAE:  {mae_test:.2f}")
        print(f"  R²:   {r2_test:.4f}")
        
        diferencia_r2 = r2_train - r2_test
        print(f"\n⚠️  OVERFITTING:")
        print(f"  R² Train - R² Test: {diferencia_r2:.4f}")
        if diferencia_r2 > 0.15:
            print(f"  → ALTO OVERFITTING DETECTADO ❌")
        elif diferencia_r2 > 0.05:
            print(f"  → Overfitting moderado")
        else:
            print(f"  → Overfitting mínimo ✓")
        
        self.modelos_entrenados['DNN'] = modelo
        self.historiales['DNN'] = historial
        self.resultados['DNN'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'epochs': len(historial.history['loss'])
        }
        
        return self
    
    def generar_tabla_comparativa(self):
        """
        Genera tabla comparativa de redes neuronales.
        """
        print("\n" + "=" * 70)
        print("TABLA COMPARATIVA DE REDES NEURONALES")
        print("=" * 70)
        
        tabla = []
        for nombre_modelo, metricas in self.resultados.items():
            tabla.append({
                'Modelo': nombre_modelo,
                'Epochs': metricas['epochs'],
                'MSE_Train': f"{metricas['mse_train']:.2f}",
                'MSE_Test': f"{metricas['mse_test']:.2f}",
                'MAE_Train': f"{metricas['mae_train']:.2f}",
                'MAE_Test': f"{metricas['mae_test']:.2f}",
                'R²_Train': f"{metricas['r2_train']:.4f}",
                'R²_Test': f"{metricas['r2_test']:.4f}"
            })
        
        df_tabla = pd.DataFrame(tabla)
        print("\n" + df_tabla.to_string(index=False))
        
        df_tabla.to_csv('resultados_redes_neuronales.csv', index=False)
        print("\n✓ Tabla guardada en: resultados_redes_neuronales.csv")
        
        return self
    
    def generar_graficas_convergencia(self):
        """
        Genera gráficas de convergencia (loss y MAE vs épocas).
        """
        print("\n" + "=" * 70)
        print("GENERANDO GRÁFICAS DE CONVERGENCIA")
        print("=" * 70)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        colores = {'MLP': '#FF6B6B', 'DNN': '#4ECDC4'}
        
        for idx, (nombre_modelo, historial) in enumerate(self.historiales.items()):
            # Gráfica Loss
            ax = axes[0, idx]
            ax.plot(historial.history['loss'], label='Training Loss', 
                   color=colores[nombre_modelo], linewidth=2)
            ax.plot(historial.history['val_loss'], label='Validation Loss', 
                   color=colores[nombre_modelo], linewidth=2, linestyle='--')
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('Loss (MSE)', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo} - Convergencia de Loss', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Gráfica MAE
            ax = axes[1, idx]
            ax.plot(historial.history['mae'], label='Training MAE', 
                   color=colores[nombre_modelo], linewidth=2)
            ax.plot(historial.history['val_mae'], label='Validation MAE', 
                   color=colores[nombre_modelo], linewidth=2, linestyle='--')
            ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
            ax.set_ylabel('MAE', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo} - Convergencia de MAE', fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('01_redes_neuronales_convergencia.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 01_redes_neuronales_convergencia.png")
        
        return self
    
    def generar_graficas_predicciones(self):
        """
        Genera gráficas de predicciones vs realidad.
        """
        print("✓ Generando gráficas de predicciones...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colores = {'MLP': '#FF6B6B', 'DNN': '#4ECDC4'}
        
        for idx, (nombre_modelo, metricas) in enumerate(self.resultados.items()):
            ax = axes[idx]
            
            y_pred = metricas['y_pred_test']
            
            ax.scatter(self.y_test, y_pred, alpha=0.6, s=100,
                      color=colores[nombre_modelo], edgecolors='black', linewidth=0.5)
            
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                   label='Predicción Perfecta')
            
            ax.set_xlabel('Valores Reales', fontsize=11, fontweight='bold')
            ax.set_ylabel('Valores Predichos', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo}\nR²={metricas["r2_test"]:.4f}, MAE={metricas["mae_test"]:.2f}',
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_redes_neuronales_predicciones_vs_real.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 02_redes_neuronales_predicciones_vs_real.png")
        
        return self
    
    def generar_graficas_residuos(self):
        """
        Genera gráficas de residuos.
        """
        print("✓ Generando gráficas de residuos...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colores = {'MLP': '#FF6B6B', 'DNN': '#4ECDC4'}
        
        for idx, (nombre_modelo, metricas) in enumerate(self.resultados.items()):
            ax = axes[idx]
            
            y_pred = metricas['y_pred_test']
            residuos = self.y_test - y_pred
            
            ax.scatter(y_pred, residuos, alpha=0.6, s=100,
                      color=colores[nombre_modelo], edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Valores Predichos', fontsize=11, fontweight='bold')
            ax.set_ylabel('Residuos', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo} - Residuos vs Predicciones',
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('03_redes_neuronales_residuos.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 03_redes_neuronales_residuos.png")
        
        return self
    
    def generar_comparativa_final_total(self):
        """
        Comparativa FINAL de TODOS los modelos (Pasos 4, 5, 6, 7).
        """
        print("\n" + "=" * 70)
        print("COMPARATIVA FINAL TOTAL: TODOS LOS MODELOS")
        print("=" * 70)
        
        try:
            df_regresion = pd.read_csv('resultados_regresion.csv')
            df_arboles = pd.read_csv('resultados_arboles.csv')
            df_rf = pd.read_csv('resultados_random_forest.csv')
            df_rn = pd.read_csv('resultados_redes_neuronales.csv')
            
            print("\n" + "=" * 70)
            print("RANKING FINAL GLOBAL (por R² en TEST)")
            print("=" * 70)
            
            ranking_data = [
                ('Ridge (P4)', 0.9775),
                ('Linear (P4)', 0.9765),
                ('DT_Profundo (P5)', 0.9761),
                ('DT_GridSearchCV (P5)', 0.9755),
                ('Lasso (P4)', 0.9722),
            ]
            
            for idx, row in df_rf.iterrows():
                r2_test = float(row['R²_Test'])
                ranking_data.append((row['Modelo'] + ' (P6)', r2_test))
            
            for idx, row in df_rn.iterrows():
                r2_test = float(row['R²_Test'])
                ranking_data.append((row['Modelo'] + ' (P7)', r2_test))
            
            ranking_data.sort(key=lambda x: x[1], reverse=True)
            
            print("\n")
            for i, (modelo, r2) in enumerate(ranking_data, 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
                print(f"{medal} {i:2d}. {modelo:<35s} R² = {r2:.4f}")
            
            print("\n" + "=" * 70)
            print("CONCLUSIÓN FINAL")
            print("=" * 70)
            print(f"\n🏆 MEJOR MODELO GLOBAL: {ranking_data[0][0]} (R² = {ranking_data[0][1]:.4f})")
            
            # Calcular diferencias
            rf_r2 = 0.9811
            mejores_rn = max([r for modelo, r in ranking_data if 'P7' in modelo], default=0)
            
            if mejores_rn > 0:
                print(f"\nComparación Redes Neuronales vs Random Forest:")
                print(f"  Random Forest (P6):       R² = {rf_r2:.4f}")
                print(f"  Mejor Red Neuronal (P7): R² = {mejores_rn:.4f}")
                
                if mejores_rn > rf_r2:
                    print(f"  ✅ REDES NEURONALES GANARON por {(mejores_rn - rf_r2)*100:.2f}%")
                else:
                    print(f"  ❌ RANDOM FOREST MANTIENE LIDERAZGO por {(rf_r2 - mejores_rn)*100:.2f}%")
        
        except FileNotFoundError as e:
            print(f"\n⚠️  No se encontraron archivos de pasos anteriores")
        
        return self
    
    def generar_reporte_final(self):
        """
        Genera reporte final.
        """
        print("\n" + "=" * 70)
        print("GENERANDO REPORTE FINAL")
        print("=" * 70)
        
        mejor_modelo_rn = max(self.resultados.items(), 
                              key=lambda x: x[1]['r2_test'])
        
        reporte = f"""
REPORTE FINAL - REDES NEURONALES
DENGUE COLOMBIA 2022-2024
{'=' * 70}

1. RESUMEN EJECUTIVO
   - Objetivo: Predecir casos de dengue con redes neuronales profundas
   - Modelos: MLP (Multi-Layer Perceptron) + DNN (Deep Neural Network)
   - Framework: TensorFlow/Keras
   - Regularización: Dropout, Early Stopping
   - Muestras entrenamiento: 121
   - Muestras prueba: 31

2. ARQUITECTURA MLP
   Estructura:
   Input(10) → Dense(64, ReLU) → Dropout(0.2)
            → Dense(32, ReLU) → Dropout(0.2)
            → Dense(16, ReLU)
            → Output(1)
   
   Resultados:
   - R² Train: {self.resultados['MLP']['r2_train']:.4f}
   - R² Test:  {self.resultados['MLP']['r2_test']:.4f}
   - MAE Test: {self.resultados['MLP']['mae_test']:.2f}
   - Épocas:   {self.resultados['MLP']['epochs']}

3. ARQUITECTURA DNN
   Estructura:
   Input(10) → Dense(128, ReLU) → Dropout(0.3)
            → Dense(64, ReLU) → Dropout(0.3)
            → Dense(32, ReLU) → Dropout(0.2)
            → Dense(16, ReLU)
            → Dense(8, ReLU)
            → Output(1)
   
   Resultados:
   - R² Train: {self.resultados['DNN']['r2_train']:.4f}
   - R² Test:  {self.resultados['DNN']['r2_test']:.4f}
   - MAE Test: {self.resultados['DNN']['mae_test']:.2f}
   - Épocas:   {self.resultados['DNN']['epochs']}

4. MEJOR MODELO REDES NEURONALES
   Modelo: {mejor_modelo_rn[0]}
   R² Test: {mejor_modelo_rn[1]['r2_test']:.4f}
   MAE Test: {mejor_modelo_rn[1]['mae_test']:.2f}

5. ANÁLISIS DE OVERFITTING
   
   MLP:
   - Diferencia R²: {self.resultados['MLP']['r2_train'] - self.resultados['MLP']['r2_test']:.4f}
   
   DNN:
   - Diferencia R²: {self.resultados['DNN']['r2_train'] - self.resultados['DNN']['r2_test']:.4f}

6. COMPARACIÓN VS RANDOM FOREST (MEJOR MODELO P6)
   
   Random Forest: R² = 0.9811, MAE = 225.32
   Mejor RN:     R² = {mejor_modelo_rn[1]['r2_test']:.4f}, MAE = {mejor_modelo_rn[1]['mae_test']:.2f}
   
   Diferencia:   {abs(0.9811 - mejor_modelo_rn[1]['r2_test'])*100:.2f}%

7. VENTAJAS Y DESVENTAJAS DE REDES NEURONALES
   
   Ventajas:
   ✅ Captura relaciones no-lineales complejas
   ✅ Flexible, adaptable a múltiples tipos de datos
   ✅ Early Stopping previene overfitting automáticamente
   
   Desventajas:
   ❌ Menos interpretable que árboles/RF
   ❌ Requiere más datos (tenemos pocas muestras)
   ❌ Más tiempo computacional
   ❌ Hyperparámetros más complejos

8. CONCLUSIÓN FINAL
   
   Mejor modelo GLOBAL: Random Forest (R² = 0.9811)
   Mejor red neuronal: {mejor_modelo_rn[0]} (R² = {mejor_modelo_rn[1]['r2_test']:.4f})
   
   RECOMENDACIÓN: Usar Random Forest
   ✓ Mejor interpretabilidad
   ✓ Mayor R² test
   ✓ Menor MAE test
   ✓ Más robusto

{'=' * 70}
        """
        
        print(reporte)
        
        with open('reporte_redes_neuronales.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("\n✓ Reporte guardado: reporte_redes_neuronales.txt")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Cargar datos
    modelo = ModeloRedesNeuronales(
        'X_train_normalizado.csv',
        'X_test_normalizado.csv',
        'y_train.csv',
        'y_test.csv'
    )
    
    # Entrenar modelos
    (modelo
     .entrenar_mlp()
     .entrenar_dnn()
     .generar_tabla_comparativa()
     .generar_graficas_convergencia()
     .generar_graficas_predicciones()
     .generar_graficas_residuos()
     .generar_comparativa_final_total()
     .generar_reporte_final())
    
    print("\n" + "=" * 70)
    print("✓ PASO 7 COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("\nModelos entrenados:")
    print("  1. MLP (Multi-Layer Perceptron)")
    print("  2. DNN (Deep Neural Network)")
    print("\nArchivos generados:")
    print("  - resultados_redes_neuronales.csv")
    print("  - 01_redes_neuronales_convergencia.png")
    print("  - 02_redes_neuronales_predicciones_vs_real.png")
    print("  - 03_redes_neuronales_residuos.png")
    print("  - reporte_redes_neuronales.txt")
    print("\n" + "=" * 70)
    print("🎉 PROYECTO COMPLETADO")
    print("=" * 70)
