import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PASO 1: LIMPIEZA DE DATOS - DENGUE COLOMBIA 2022-2024
# ============================================================================
# Este script realiza la limpieza y preprocesamiento de datos de dengue
# de los archivos: Datos_2022_210.xlsx, Datos_2023_210.xlsx, Datos_2024_210.xlsx
# ============================================================================

class LimpiadorDengue:
    """
    Clase para limpiar y preprocesar datos de dengue en Colombia.
    """
    
    def __init__(self, rutas_archivos):
        """
        Inicializa la clase con las rutas de los archivos.
        
        Parámetros:
        - rutas_archivos: lista de rutas a los archivos Excel
        """
        self.rutas_archivos = rutas_archivos
        self.df_original = None
        self.df_limpio = None
        self.estadisticas_limpieza = {}
        
    def cargar_datos(self):
        """
        Carga los datos de los archivos Excel y los concatena.
        """
        print("=" * 70)
        print("CARGANDO DATOS DE DENGUE...")
        print("=" * 70)
        
        dfs = []
        for ruta in self.rutas_archivos:
            try:
                df = pd.read_excel(ruta)
                dfs.append(df)
                print(f"✓ Cargado: {ruta} - {df.shape[0]} registros, {df.shape[1]} columnas")
            except Exception as e:
                print(f"✗ Error cargando {ruta}: {e}")
                
        self.df_original = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal de registros cargados: {len(self.df_original):,}")
        print(f"Rango de años: {self.df_original['ANO'].min()} - {self.df_original['ANO'].max()}")
        
        return self
    
    def analizar_valores_faltantes(self):
        """
        Analiza y reporta los valores faltantes en el dataset.
        """
        print("\n" + "=" * 70)
        print("ANÁLISIS DE VALORES FALTANTES")
        print("=" * 70)
        
        valores_faltantes = self.df_original.isnull().sum()
        porcentaje_faltantes = (valores_faltantes / len(self.df_original) * 100).round(2)
        
        df_faltantes = pd.DataFrame({
            'Columna': valores_faltantes.index,
            'Cantidad': valores_faltantes.values,
            'Porcentaje': porcentaje_faltantes.values
        }).sort_values('Porcentaje', ascending=False)
        
        df_faltantes = df_faltantes[df_faltantes['Cantidad'] > 0]
        
        if len(df_faltantes) > 0:
            print("\nValores faltantes encontrados:")
            print(df_faltantes.to_string(index=False))
        else:
            print("\n✓ No hay valores faltantes en el dataset.")
        
        self.estadisticas_limpieza['valores_faltantes'] = df_faltantes
        return self
    
    def seleccionar_columnas_relevantes(self):
        """
        Selecciona solo las columnas relevantes para el análisis de predicción semanal.
        Excluye variables del sistema SivigILA web 4.0 que no aportan valor.
        """
        print("\n" + "=" * 70)
        print("SELECCIONANDO COLUMNAS RELEVANTES")
        print("=" * 70)
        
        # Columnas relevantes para análisis semanal de casos de dengue
        columnas_relevantes = [
            'SEMANA',           # Variable objetivo temporal
            'ANO',              # Año
            'EDAD',             # Edad del paciente
            'SEXO',             # Sexo del paciente
            'COD_DPTO_O',       # Código departamento de ocurrencia
            'COD_MUN_O',        # Código municipio de ocurrencia
            'AREA',             # Área urbana/rural
            'TIP_SS',           # Tipo seguridad social
            'TIP_CAS',          # Tipo de caso
            'Estado_final_de_caso',  # Estado final
            'confirmados',      # Confirmado (target para algunos análisis)
            'FEC_NOT',          # Fecha de notificación
            'FEC_CON',          # Fecha de confirmación
        ]
        
        # Validar que las columnas existan
        columnas_disponibles = [col for col in columnas_relevantes if col in self.df_original.columns]
        columnas_faltantes = [col for col in columnas_relevantes if col not in self.df_original.columns]
        
        if columnas_faltantes:
            print(f"⚠ Columnas solicitadas no encontradas: {columnas_faltantes}")
        
        self.df_original = self.df_original[columnas_disponibles].copy()
        print(f"\n✓ Columnas seleccionadas: {len(columnas_disponibles)}")
        print(f"Columnas: {columnas_disponibles}")
        
        return self
    
    def limpiar_valores_atipicos(self):
        """
        Detecta y maneja valores atípicos en variables numéricas.
        """
        print("\n" + "=" * 70)
        print("DETECTANDO VALORES ATÍPICOS")
        print("=" * 70)
        
        # Edad: debe estar entre 0 y 120
        edad_invalida = self.df_original[(self.df_original['EDAD'] < 0) | 
                                        (self.df_original['EDAD'] > 120)].shape[0]
        if edad_invalida > 0:
            print(f"⚠ Edades inválidas encontradas: {edad_invalida}")
            self.df_original = self.df_original[(self.df_original['EDAD'] >= 0) & 
                                              (self.df_original['EDAD'] <= 120)]
        
        # Semana: debe estar entre 1 y 53
        semana_invalida = self.df_original[(self.df_original['SEMANA'] < 1) | 
                                          (self.df_original['SEMANA'] > 53)].shape[0]
        if semana_invalida > 0:
            print(f"⚠ Semanas inválidas encontradas: {semana_invalida}")
            self.df_original = self.df_original[(self.df_original['SEMANA'] >= 1) & 
                                              (self.df_original['SEMANA'] <= 53)]
        
        # Año: debe estar entre 2022 y 2024
        ano_invalido = self.df_original[(self.df_original['ANO'] < 2022) | 
                                       (self.df_original['ANO'] > 2024)].shape[0]
        if ano_invalido > 0:
            print(f"⚠ Años inválidos encontrados: {ano_invalido}")
            self.df_original = self.df_original[(self.df_original['ANO'] >= 2022) & 
                                              (self.df_original['ANO'] <= 2024)]
        
        print("✓ Limpieza de valores atípicos completada.")
        
        return self
    
    def crear_variable_objetivo(self):
        """
        Crea la variable objetivo para la predicción: total de casos por semana.
        """
        print("\n" + "=" * 70)
        print("CREANDO VARIABLE OBJETIVO (CASOS POR SEMANA)")
        print("=" * 70)
        
        # Crear tabla agregada: casos por semana y año
        casos_por_semana = self.df_original.groupby(['ANO', 'SEMANA']).size().reset_index(name='TOTAL_CASOS')
        
        print(f"\n✓ Agregación completada.")
        print(f"Total de semanas registradas: {len(casos_por_semana)}")
        print(f"\nEstadísticas de TOTAL_CASOS:")
        print(f"  - Media: {casos_por_semana['TOTAL_CASOS'].mean():.2f}")
        print(f"  - Mediana: {casos_por_semana['TOTAL_CASOS'].median():.2f}")
        print(f"  - Desv. Estándar: {casos_por_semana['TOTAL_CASOS'].std():.2f}")
        print(f"  - Mínimo: {casos_por_semana['TOTAL_CASOS'].min()}")
        print(f"  - Máximo: {casos_por_semana['TOTAL_CASOS'].max()}")
        
        # Guardar para análisis posterior
        casos_por_semana.to_csv('casos_por_semana_agregado.csv', index=False)
        print("\n✓ Datos agregados guardados en 'casos_por_semana_agregado.csv'")
        
        return casos_por_semana
    
    def crear_features_temporales(self, df_agregado):
        """
        Crea características temporales para mejorar la predicción.
        """
        print("\n" + "=" * 70)
        print("CREANDO CARACTERÍSTICAS TEMPORALES")
        print("=" * 70)
        
        df_features = df_agregado.copy()
        
        # Feature 1: Crear identificador único temporal
        df_features['SEMANA_TEMPORAL'] = df_features['ANO'] * 100 + df_features['SEMANA']
        
        # Feature 2: Período del año (cuartil)
        df_features['PERIODO_ANIO'] = pd.cut(df_features['SEMANA'], 
                                            bins=[0, 13, 26, 39, 53], 
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'])
        
        # Feature 3: Estación aproximada (Colombia)
        # Lluviosa: abril-junio (semanas 14-26) y octubre-noviembre (semanas 41-45)
        df_features['ES_LLUVIOSA'] = df_features['SEMANA'].apply(
            lambda x: 1 if (14 <= x <= 26) or (41 <= x <= 45) else 0
        )
        
        print("\n✓ Features temporales creadas:")
        print(f"  - SEMANA_TEMPORAL: identificador único temporal")
        print(f"  - PERIODO_ANIO: cuartil del año")
        print(f"  - ES_LLUVIOSA: 1 si es época lluviosa, 0 si no")
        
        df_features.to_csv('datos_dengue_con_features.csv', index=False)
        print("\n✓ Datos con features guardados en 'datos_dengue_con_features.csv'")
        
        return df_features
    
    def generar_reporte_limpieza(self):
        """
        Genera un reporte final de la limpieza.
        """
        print("\n" + "=" * 70)
        print("REPORTE FINAL DE LIMPIEZA")
        print("=" * 70)
        
        reporte = f"""
DATASET LIMPIO - DENGUE COLOMBIA 2022-2024
{'=' * 70}

Registros totales después de limpieza: {len(self.df_original):,}
Columnas: {len(self.df_original.columns)}
Rango temporal: {self.df_original['ANO'].min()} - {self.df_original['ANO'].max()}
Semanas: 1-53

Distribución por año:
{self.df_original['ANO'].value_counts().sort_index().to_string()}

Distribución por sexo:
{self.df_original['SEXO'].value_counts(dropna=False).to_string()}

Estadísticas de edad:
{self.df_original['EDAD'].describe().to_string()}

Archivos generados:
1. casos_por_semana_agregado.csv - Casos totales por semana
2. datos_dengue_con_features.csv - Datos con características temporales
        """
        
        print(reporte)
        
        # Guardar reporte
        with open('reporte_limpieza_dengue.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        return reporte


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Rutas de los archivos (CAMBIAR SEGÚN UBICACIÓN LOCAL)
    rutas = [
        'Datos_2022_210.xlsx',
        'Datos_2023_210.xlsx',
        'Datos_2024_210.xlsx'
    ]
    
    # Crear instancia del limpiador
    limpiador = LimpiadorDengue(rutas)
    
    # Ejecutar pipeline de limpieza
    limpiador.cargar_datos()\
             .analizar_valores_faltantes()\
             .seleccionar_columnas_relevantes()\
             .limpiar_valores_atipicos()
    
    # Crear variable objetivo
    df_agregado = limpiador.crear_variable_objetivo()
    
    # Crear features temporales
    df_features = limpiador.crear_features_temporales(df_agregado)
    
    # Generar reporte
    limpiador.generar_reporte_limpieza()
    
    print("\n✓ PASO 1 COMPLETADO EXITOSAMENTE")
    print("Los archivos están listos para el siguiente paso: Exploración (EDA)")
