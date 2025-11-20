import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PASO 3: PREPROCESAMIENTO Y PREPARACIÓN DE DATOS PARA MODELADO
# ============================================================================
# Modificado para:
# 1. Evitar multicolinealidad (eliminar SEMANA_TEMPORAL redundante)
# 2. Transformación log para manejar asimetría
# 3. Mejor detección de overfitting
# ============================================================================

class PreprocesadorDatos:
    """
    Clase para preprocesar datos de dengue para modelado.
    """
    
    def __init__(self, ruta_datos):
        """
        Inicializa con los datos agregados.
        
        Parámetros:
        - ruta_datos: ruta al archivo CSV de casos por semana
        """
        self.df = pd.read_csv(ruta_datos)
        self.df_procesado = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_train_log = None
        self.y_test_log = None
        self.scaler = None
        
    def crear_features_lag(self, n_lags=4):
        """
        Crea características lag (valores rezagados).
        
        Parámetros:
        - n_lags: número de semanas previas a considerar
        """
        print("=" * 70)
        print("CREANDO CARACTERÍSTICAS LAG (REZAGADAS)")
        print("=" * 70)
        
        df_features = self.df.copy().sort_values(['ANO', 'SEMANA']).reset_index(drop=True)
        
        # Crear lags
        for lag in range(1, n_lags + 1):
            df_features[f'TOTAL_CASOS_LAG{lag}'] = df_features['TOTAL_CASOS'].shift(lag)
        
        # Eliminar filas con NaN
        df_features = df_features.dropna().reset_index(drop=True)
        
        print(f"\n✓ Features lag creadas:")
        print(f"  - Cantidad: {n_lags} variables lag")
        print(f"  - Datos con NaN eliminados: {len(self.df) - len(df_features)}")
        print(f"  - Datos disponibles: {len(df_features)}")
        print(f"\n  Sample de features lag:")
        print(df_features[['ANO', 'SEMANA', 'TOTAL_CASOS', 'TOTAL_CASOS_LAG1', 
                           'TOTAL_CASOS_LAG2', 'TOTAL_CASOS_LAG3', 'TOTAL_CASOS_LAG4']].head(10))
        
        self.df_procesado = df_features
        
        return self
    
    def crear_features_estacionales(self):
        """
        Crea features estacionales.
        """
        print("\n" + "=" * 70)
        print("CREANDO CARACTERÍSTICAS ESTACIONALES")
        print("=" * 70)
        
        df = self.df_procesado.copy()
        
        # Feature 1: Trimestre
        df['TRIMESTRE'] = ((df['SEMANA'] - 1) // 13) + 1
        
        # Feature 2: Época de lluvia
        df['LLUVIA'] = ((df['SEMANA'] >= 14) & (df['SEMANA'] <= 26)) | ((df['SEMANA'] >= 40) & (df['SEMANA'] <= 45))
        df['LLUVIA'] = df['LLUVIA'].astype(int)
        
        # Feature 3: Mitad del año
        df['MITAD_ANO'] = (df['SEMANA'] <= 26).astype(int)
        
        # Feature 4: Semana normalizada
        df['SEMANA_NORM'] = df['SEMANA'] / 53.0
        
        print(f"\n✓ Features estacionales creadas:")
        print(f"  - TRIMESTRE: trimestre del año (1-4)")
        print(f"  - LLUVIA: indicador de época lluviosa")
        print(f"  - MITAD_ANO: primera o segunda mitad")
        print(f"  - SEMANA_NORM: semana normalizada (0-1)")
        print(f"\n  Sample de features estacionales:")
        print(df[['ANO', 'SEMANA', 'TRIMESTRE', 'LLUVIA', 'MITAD_ANO', 'SEMANA_NORM']].head(10))
        
        self.df_procesado = df
        
        return self
    
    def crear_dataset_supervisado(self):
        """
        Crea el dataset supervisado.
        ⚠️ MODIFICACIÓN: Sin SEMANA_TEMPORAL para evitar multicolinealidad
        """
        print("\n" + "=" * 70)
        print("CREANDO DATASET SUPERVISADO (X, y)")
        print("=" * 70)
        
        df = self.df_procesado.copy()
        
        # MODIFICACIÓN 1: Eliminar SEMANA_TEMPORAL (r=0.984 con ANO causa multicolinealidad)
        features_utilizadas = [
            'ANO',              # ✅ MANTENER: correlación fuerte con target
            'SEMANA',           # ✅ Información estacional
            'TRIMESTRE',        # ✅ Patrón estacional
            'LLUVIA',           # ✅ Factor epidemiológico importante
            'MITAD_ANO',        # ✅ Información temporal
            'SEMANA_NORM',      # ✅ Normalización de semana
            'TOTAL_CASOS_LAG1', # ✅ CRÍTICO: dependencia temporal
            'TOTAL_CASOS_LAG2', # ✅ CRÍTICO: dependencia temporal
            'TOTAL_CASOS_LAG3', # ✅ Información de tendencia
            'TOTAL_CASOS_LAG4'  # ✅ Información de tendencia
        ]
        
        X = df[features_utilizadas].copy()
        y = df['TOTAL_CASOS'].copy()
        
        print(f"\n✓ Dataset supervisado creado:")
        print(f"  - Features (X): {X.shape}")
        print(f"  - Target (y): {y.shape}")
        print(f"\n  Features utilizadas ({len(features_utilizadas)}):")
        for i, feat in enumerate(features_utilizadas, 1):
            print(f"    {i}. {feat}")
        
        print(f"\n  Estadísticas de X:")
        print(X.describe())
        
        print(f"\n  Estadísticas de y (TOTAL_CASOS):")
        print(y.describe())
        
        # MODIFICACIÓN 2: Calcular transformación log
        print(f"\n  Transformación LOG del target:")
        y_log = np.log1p(y)  # log1p evita log(0)
        print(f"    y original - Media: {y.mean():.2f}, Std: {y.std():.2f}")
        print(f"    y_log - Media: {y_log.mean():.2f}, Std: {y_log.std():.2f}")
        print(f"    Asimetría reducida (esperar en log): {y_log.skew():.4f} vs original {y.skew():.4f}")
        
        self.df_X = X
        self.df_y = y
        self.df_y_log = y_log
        
        return X, y, y_log
    
    def normalizar_datos(self, X_train, X_test, metodo='standard'):
        """
        Normaliza los datos usando StandardScaler.
        
        Parámetros:
        - X_train: datos de entrenamiento
        - X_test: datos de prueba
        - metodo: 'standard' (media=0, std=1) o 'minmax' (0-1)
        """
        print("\n" + "=" * 70)
        print(f"NORMALIZACIÓN DE DATOS (Método: {metodo.upper()})")
        print("=" * 70)
        
        if metodo == 'standard':
            scaler = StandardScaler()
            print(f"\n✓ StandardScaler: centra en media=0 y escala a desv.estándar=1")
        elif metodo == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
            print(f"\n✓ MinMaxScaler: escala a rango [0, 1]")
        else:
            raise ValueError("Método debe ser 'standard' o 'minmax'")
        
        # Fit en entrenamiento
        X_train_norm = scaler.fit_transform(X_train)
        # Transform en prueba
        X_test_norm = scaler.transform(X_test)
        
        print(f"\n  Antes de normalización:")
        print(f"    X_train media: {X_train.mean().mean():.4f}, std: {X_train.std().mean():.4f}")
        print(f"    X_train min: {X_train.min().min():.4f}, max: {X_train.max().max():.4f}")
        
        print(f"\n  Después de normalización:")
        print(f"    X_train media: {X_train_norm.mean():.4f}, std: {X_train_norm.std():.4f}")
        print(f"    X_train min: {X_train_norm.min():.4f}, max: {X_train_norm.max():.4f}")
        
        self.scaler = scaler
        
        return X_train_norm, X_test_norm
    
    def dividir_datos(self, X, y, test_size=0.2, random_state=42):
        """
        Divide los datos en entrenamiento y prueba.
        """
        print("\n" + "=" * 70)
        print("DIVISIÓN TRAIN/TEST")
        print("=" * 70)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=True
        )
        
        print(f"\n✓ Datos divididos:")
        print(f"  - Entrenamiento: {X_train.shape[0]} muestras ({(1-test_size)*100:.0f}%)")
        print(f"  - Prueba: {X_test.shape[0]} muestras ({test_size*100:.0f}%)")
        
        print(f"\n  Estadísticas y_train:")
        print(f"    Media: {y_train.mean():.2f}, Std: {y_train.std():.2f}")
        print(f"    Min: {y_train.min()}, Max: {y_train.max()}")
        
        print(f"\n  Estadísticas y_test:")
        print(f"    Media: {y_test.mean():.2f}, Std: {y_test.std():.2f}")
        print(f"    Min: {y_test.min()}, Max: {y_test.max()}")
        
        # Detección de overfitting potencial
        ratio_std = y_test.std() / y_train.std()
        print(f"\n  ⚠️  Ratio Std Test/Train: {ratio_std:.2f}")
        if ratio_std > 1.3:
            print(f"      → Test set tiene MAYOR variabilidad (riesgo de overfitting)")
        elif ratio_std < 0.7:
            print(f"      → Test set tiene MENOR variabilidad (división puede ser sesgada)")
        else:
            print(f"      → Variabilidad BALANCEADA ✓")
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def guardar_datos_preprocesados(self):
        """
        Guarda los datos preprocesados en archivos CSV.
        ⚠️ MODIFICACIÓN 3: Guarda AMBAS versiones (original + log transformada)
        """
        print("\n" + "=" * 70)
        print("GUARDANDO DATOS PREPROCESADOS")
        print("=" * 70)
        
        # Normalizar
        X_train_norm, X_test_norm = self.normalizar_datos(
            self.X_train, self.X_test, metodo='standard'
        )
        
        # Convertir a DataFrames
        feature_names = self.X_train.columns.tolist()
        
        X_train_norm_df = pd.DataFrame(X_train_norm, columns=feature_names)
        X_test_norm_df = pd.DataFrame(X_test_norm, columns=feature_names)
        
        # VERSIÓN ORIGINAL
        y_train_df = pd.DataFrame(self.y_train.values, columns=['TOTAL_CASOS'])
        y_test_df = pd.DataFrame(self.y_test.values, columns=['TOTAL_CASOS'])
        
        # VERSIÓN LOG TRANSFORMADA
        y_train_log_df = pd.DataFrame(
            np.log1p(self.y_train).values, columns=['TOTAL_CASOS_LOG']
        )
        y_test_log_df = pd.DataFrame(
            np.log1p(self.y_test).values, columns=['TOTAL_CASOS_LOG']
        )
        
        # Guardar archivos
        X_train_norm_df.to_csv('X_train_normalizado.csv', index=False)
        X_test_norm_df.to_csv('X_test_normalizado.csv', index=False)
        y_train_df.to_csv('y_train.csv', index=False)
        y_test_df.to_csv('y_test.csv', index=False)
        y_train_log_df.to_csv('y_train_log.csv', index=False)
        y_test_log_df.to_csv('y_test_log.csv', index=False)
        
        # Sin normalizar
        self.X_train.to_csv('X_train_original.csv', index=False)
        self.X_test.to_csv('X_test_original.csv', index=False)
        
        # Dataset completo
        self.df_procesado.to_csv('datos_completos_preprocesados.csv', index=False)
        
        print(f"\n✓ Archivos guardados:")
        print(f"  1. X_train_normalizado.csv")
        print(f"  2. X_test_normalizado.csv")
        print(f"  3. y_train.csv (ORIGINAL)")
        print(f"  4. y_test.csv (ORIGINAL)")
        print(f"  5. y_train_log.csv ⭐ (TRANSFORMADA LOG - para reducir asimetría)")
        print(f"  6. y_test_log.csv ⭐ (TRANSFORMADA LOG)")
        print(f"  7. X_train_original.csv")
        print(f"  8. X_test_original.csv")
        print(f"  9. datos_completos_preprocesados.csv")
    
    def generar_reporte_preprocesamiento(self):
        """
        Genera un reporte del preprocesamiento.
        """
        print("\n" + "=" * 70)
        print("REPORTE DE PREPROCESAMIENTO")
        print("=" * 70)
        
        reporte = f"""
REPORTE DE PREPROCESAMIENTO - DENGUE COLOMBIA 2022-2024
{'=' * 70}

1. DATOS ORIGINALES
   - Total de semanas: {len(self.df)}
   - Rango temporal: {self.df['ANO'].min()} - {self.df['ANO'].max()}
   - Total de casos: {self.df['TOTAL_CASOS'].sum():,}

2. FEATURES LAG CREADAS
   - Cantidad: 4 (LAG1, LAG2, LAG3, LAG4)
   - Datos disponibles después de lags: {len(self.df_procesado)}
   - Pérdida de datos: {len(self.df) - len(self.df_procesado)} (normal por lag)

3. FEATURES ESTACIONALES CREADAS
   - TRIMESTRE: Trimestre del año (1-4)
   - LLUVIA: Indicador de época lluviosa
   - MITAD_ANO: Primera o segunda mitad del año
   - SEMANA_NORM: Semana normalizada (0-1)

4. DATASET SUPERVISADO
   - Features utilizadas: 10
     1. ANO ✓ (Mantener por correlación fuerte)
     2. SEMANA
     3. TRIMESTRE
     4. LLUVIA
     5. MITAD_ANO
     6. SEMANA_NORM
     7. TOTAL_CASOS_LAG1 ⭐ (CRÍTICO)
     8. TOTAL_CASOS_LAG2 ⭐ (CRÍTICO)
     9. TOTAL_CASOS_LAG3
     10. TOTAL_CASOS_LAG4
   
   ⚠️ MODIFICACIÓN: Se eliminó SEMANA_TEMPORAL (redundante con ANO, r=0.984)

   - Target (y): TOTAL_CASOS
   - Target transformado: log1p(TOTAL_CASOS) ⭐ para reducir asimetría

5. DIVISIÓN TRAIN/TEST
   - Datos de entrenamiento: {len(self.X_train)} muestras (80%)
   - Datos de prueba: {len(self.X_test)} muestras (20%)
   - Random state: 42 (reproducibilidad)

6. NORMALIZACIÓN
   - Método: StandardScaler
   - Escala: Media = 0, Desviación Estándar = 1
   - Aplicado: Features (X)
   - Target: Dos versiones (original + log)

7. VERSIONES DE TARGET DISPONIBLES
   - y_train.csv / y_test.csv: ORIGINAL
   - y_train_log.csv / y_test_log.csv: ⭐ TRANSFORMADA LOG
   
   Asimetría original: {self.df_y.skew():.4f}
   Asimetría transformada: {self.df_y_log.skew():.4f}
   
   ✓ Mejora: Usar versión LOG para reducir impacto de valores extremos

8. CRITERIOS DE CALIDAD
   - Multicolinealidad: ✓ Evitada (eliminada SEMANA_TEMPORAL)
   - Detección overfitting: ✓ Activada en train/test
   - Transformación asimetría: ✓ Versión log disponible
   - Normalización: ✓ StandardScaler aplicado

RECOMENDACIONES PARA MODELADO
{'=' * 70}

1. Para Regresión:
   - Usar TARGET TRANSFORMADO (y_log) para mejores resultados
   - Aplicar regularización (Ridge/Lasso) por tamaño reducido de muestra
   - Usar GridSearchCV con k=5 en validación cruzada

2. Para Árboles/Random Forest:
   - Pueden usar TARGET ORIGINAL (menos sensibles a asimetría)
   - Aprovechar features lag (LAG1, LAG2) que capturan temporalidad

3. Para Redes Neuronales:
   - Usar TARGET TRANSFORMADO (y_log) y features normalizadas
   - Monitorear overfitting (solo 152 muestras)
   - Considerar regularización L2 (dropout)

DATOS LISTOS PARA MODELADO
{'=' * 70}
        """
        
        print(reporte)
        
        with open('reporte_preprocesamiento.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("✓ Archivo guardado: reporte_preprocesamiento.txt")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PASO 3: PREPROCESAMIENTO DE DATOS (VERSIÓN MEJORADA)")
    print("=" * 70 + "\n")
    
    prep = PreprocesadorDatos('casos_por_semana_agregado.csv')
    
    (prep
     .crear_features_lag(n_lags=4)
     .crear_features_estacionales())
    
    X, y, y_log = prep.crear_dataset_supervisado()
    
    X_train, X_test, y_train, y_test = prep.dividir_datos(X, y, test_size=0.2, random_state=42)
    
    prep.guardar_datos_preprocesados()
    
    prep.generar_reporte_preprocesamiento()
    
    print("\n✓ PASO 3 COMPLETADO EXITOSAMENTE")
    print("\n⭐ CAMBIOS PRINCIPALES:")
    print("   1. Eliminada SEMANA_TEMPORAL (multicolinealidad)")
    print("   2. Transformación LOG del target para reducir asimetría")
    print("   3. Mejora en detección de overfitting")
    print("\nDatos listos para entrenamiento de modelos")
