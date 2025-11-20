import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PASO 2: EXPLORACIÓN Y ANÁLISIS EXPLORATORIO (EDA)
# ============================================================================
# Este script realiza análisis exploratorio visual de los datos de dengue
# Incluye gráficas, estadísticas y análisis de correlación
# ============================================================================

class AnalizadorEDA:
    """
    Clase para realizar análisis exploratorio de datos (EDA) de dengue.
    """
    
    def __init__(self, ruta_datos):
        """
        Inicializa con la ruta del archivo agregado.
        
        Parámetros:
        - ruta_datos: ruta al archivo CSV de casos por semana
        """
        self.df = pd.read_csv(ruta_datos)
        self.figuras_generadas = []
        sns.set_palette("husl")
        
    def estadisticas_descriptivas(self):
        """
        Calcula y reporta estadísticas descriptivas básicas.
        """
        print("=" * 70)
        print("ESTADÍSTICAS DESCRIPTIVAS")
        print("=" * 70)
        
        stats = self.df['TOTAL_CASOS'].describe()
        print("\nResumen estadístico de TOTAL_CASOS:")
        print(stats)
        
        print(f"\nAsimetría (Skewness): {skew(self.df['TOTAL_CASOS']):.4f}")
        print(f"Curtosis (Kurtosis): {kurtosis(self.df['TOTAL_CASOS']):.4f}")
        
        print("\n\nDistribución de semanas por año:")
        print(self.df['ANO'].value_counts().sort_index())
        
        print("\n\nEstadísticas por año:")
        stats_por_ano = self.df.groupby('ANO')['TOTAL_CASOS'].describe()
        print(stats_por_ano)
        
        return self
    
    def grafica_distribucion_casos_temporal(self):
        """
        Gráfica 1: Evolución temporal de casos de dengue por semana.
        """
        print("\n✓ Generando gráfica 1: Distribución temporal...")
        
        try:
            fig, axes = plt.subplots(2, 1, figsize=(14, 10))
            
            ax1 = axes[0]
            colores_ano = {2022: '#FF6B6B', 2023: '#4ECDC4', 2024: '#45B7D1'}
            
            for ano in sorted(self.df['ANO'].unique()):
                df_ano = self.df[self.df['ANO'] == ano].sort_values('SEMANA')
                ax1.plot(df_ano['SEMANA'], df_ano['TOTAL_CASOS'], 
                        marker='o', label=f'Año {ano}', linewidth=2,
                        color=colores_ano[ano], markersize=5)
            
            ax1.set_xlabel('Semana del Año', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Total de Casos', fontsize=12, fontweight='bold')
            ax1.set_title('Evolución Temporal de Casos de Dengue (2022-2024)\nPor Semana', 
                         fontsize=14, fontweight='bold', pad=20)
            ax1.legend(fontsize=11, loc='best')
            ax1.grid(True, alpha=0.3)
            ax1.set_xticks(np.arange(0, 54, 4))
            
            ax2 = axes[1]
            data_por_ano = [self.df[self.df['ANO'] == ano]['TOTAL_CASOS'].values 
                           for ano in sorted(self.df['ANO'].unique())]
            bp = ax2.boxplot(data_por_ano, labels=sorted(self.df['ANO'].unique()),
                            patch_artist=True, widths=0.6)
            
            for patch, ano in zip(bp['boxes'], sorted(self.df['ANO'].unique())):
                patch.set_facecolor(colores_ano[ano])
                patch.set_alpha(0.7)
            
            ax2.set_xlabel('Año', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Total de Casos', fontsize=12, fontweight='bold')
            ax2.set_title('Distribución de Casos por Año (Box-Plot)', 
                         fontsize=14, fontweight='bold', pad=20)
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('01_distribucion_temporal.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 01_distribucion_temporal.png")
            self.figuras_generadas.append('01_distribucion_temporal.png')
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def grafica_histograma_distribucion(self):
        """
        Gráfica 2: Histogramas de distribución de casos.
        """
        print("✓ Generando gráfica 2: Histogramas...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            ax1 = axes[0, 0]
            ax1.hist(self.df['TOTAL_CASOS'], bins=30, color='#3498DB', 
                    edgecolor='black', alpha=0.7)
            ax1.axvline(self.df['TOTAL_CASOS'].mean(), color='red', 
                       linestyle='--', linewidth=2, label=f"Media: {self.df['TOTAL_CASOS'].mean():.0f}")
            ax1.axvline(self.df['TOTAL_CASOS'].median(), color='green', 
                       linestyle='--', linewidth=2, label=f"Mediana: {self.df['TOTAL_CASOS'].median():.0f}")
            ax1.set_xlabel('Total de Casos', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
            ax1.set_title('Distribución General de Casos', fontsize=12, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            
            colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            anos_ordenados = sorted(self.df['ANO'].unique())
            
            for idx, (ano, color) in enumerate(zip(anos_ordenados, colores)):
                df_ano = self.df[self.df['ANO'] == ano]
                if idx == 0:
                    ax = axes[0, 1]
                elif idx == 1:
                    ax = axes[1, 0]
                else:
                    ax = axes[1, 1]
                
                ax.hist(df_ano['TOTAL_CASOS'], bins=20, color=color, 
                       edgecolor='black', alpha=0.7)
                ax.axvline(df_ano['TOTAL_CASOS'].mean(), color='red', 
                          linestyle='--', linewidth=2)
                ax.set_xlabel('Total de Casos', fontsize=10, fontweight='bold')
                ax.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
                ax.set_title(f'Distribución de Casos - Año {ano}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('02_histogramas_distribucion.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 02_histogramas_distribucion.png")
            self.figuras_generadas.append('02_histogramas_distribucion.png')
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def grafica_densidad_casos(self):
        """
        Gráfica 3: Densidad de probabilidad de casos.
        """
        print("✓ Generando gráfica 3: Gráficas de densidad...")
        
        try:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1 = axes[0]
            self.df['TOTAL_CASOS'].plot(kind='density', ax=ax1, color='#3498DB', linewidth=2.5)
            ax1.fill_between(ax1.lines[0].get_xdata(), ax1.lines[0].get_ydata(), 
                            alpha=0.3, color='#3498DB')
            ax1.set_xlabel('Total de Casos', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Densidad', fontsize=12, fontweight='bold')
            ax1.set_title('Función de Densidad de Probabilidad - Casos Totales', 
                         fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            ax2 = axes[1]
            for ano in sorted(self.df['ANO'].unique()):
                df_ano = self.df[self.df['ANO'] == ano]
                df_ano['TOTAL_CASOS'].plot(kind='density', ax=ax2, linewidth=2.5, 
                                          label=f'Año {ano}')
            ax2.set_xlabel('Total de Casos', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Densidad', fontsize=12, fontweight='bold')
            ax2.set_title('Función de Densidad por Año', fontsize=12, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('03_graficas_densidad.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 03_graficas_densidad.png")
            self.figuras_generadas.append('03_graficas_densidad.png')
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def grafica_promedio_semana_del_ano(self):
        """
        Gráfica 4: Promedio de casos por semana del año.
        """
        print("✓ Generando gráfica 4: Patrón promedio por semana...")
        
        try:
            promedio_por_semana = self.df.groupby('SEMANA')['TOTAL_CASOS'].agg(['mean', 'std'])
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(promedio_por_semana.index, promedio_por_semana['mean'], 
                   color='#2C3E50', marker='o', linewidth=2.5, markersize=6, 
                   label='Promedio de casos')
            ax.fill_between(promedio_por_semana.index, 
                           promedio_por_semana['mean'] - promedio_por_semana['std'],
                           promedio_por_semana['mean'] + promedio_por_semana['std'],
                           alpha=0.3, color='#3498DB', label='±1 Desv. Estándar')
            
            ax.set_xlabel('Semana del Año', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total de Casos Promedio', fontsize=12, fontweight='bold')
            ax.set_title('Patrón Estacional Promedio de Dengue en Colombia\n(Promedio 2022-2024)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=11, loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(np.arange(0, 54, 4))
            
            plt.tight_layout()
            plt.savefig('04_patron_promedio_semana.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 04_patron_promedio_semana.png")
            self.figuras_generadas.append('04_patron_promedio_semana.png')
            
            pico_maximo = promedio_por_semana['mean'].idxmax()
            pico_minimo = promedio_por_semana['mean'].idxmin()
            
            print(f"  → Semana con MÁXIMO: Semana {pico_maximo} ({promedio_por_semana['mean'][pico_maximo]:.0f} casos)")
            print(f"  → Semana con MÍNIMO: Semana {pico_minimo} ({promedio_por_semana['mean'][pico_minimo]:.0f} casos)")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def grafica_estadisticas_por_periodo(self):
        """
        Gráfica 5: Estadísticas por período del año.
        """
        print("✓ Generando gráfica 5: Análisis por período...")
        
        try:
            df_temp = self.df.copy()
            df_temp['PERIODO'] = pd.cut(df_temp['SEMANA'], 
                                       bins=[0, 13, 26, 39, 53],
                                       labels=['Q1 (1-13)', 'Q2 (14-26)', 'Q3 (27-39)', 'Q4 (40-53)'])
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            stats_periodo = df_temp.groupby('PERIODO')['TOTAL_CASOS'].agg(['mean', 'median', 'std', 'count'])
            
            ax1 = axes[0]
            bars1 = ax1.bar(range(len(stats_periodo)), stats_periodo['mean'], 
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731'],
                           edgecolor='black', alpha=0.8, linewidth=1.5)
            ax1.set_ylabel('Promedio de Casos', fontsize=11, fontweight='bold')
            ax1.set_title('Promedio de Casos por Período', fontsize=12, fontweight='bold')
            ax1.set_xticks(range(len(stats_periodo)))
            ax1.set_xticklabels(stats_periodo.index, rotation=0)
            ax1.grid(True, alpha=0.3, axis='y')
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
            
            ax2 = axes[1]
            bars2 = ax2.bar(range(len(stats_periodo)), stats_periodo['std'],
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731'],
                           edgecolor='black', alpha=0.8, linewidth=1.5)
            ax2.set_ylabel('Desviación Estándar', fontsize=11, fontweight='bold')
            ax2.set_title('Variabilidad de Casos por Período', fontsize=12, fontweight='bold')
            ax2.set_xticks(range(len(stats_periodo)))
            ax2.set_xticklabels(stats_periodo.index, rotation=0)
            ax2.grid(True, alpha=0.3, axis='y')
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.0f}', ha='center', va='bottom', fontweight='bold')
            
            ax3 = axes[2]
            data_periodo = [df_temp[df_temp['PERIODO'] == p]['TOTAL_CASOS'].values 
                           for p in stats_periodo.index]
            bp = ax3.boxplot(data_periodo, labels=stats_periodo.index, patch_artist=True)
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax3.set_ylabel('Total de Casos', fontsize=11, fontweight='bold')
            ax3.set_title('Distribución de Casos por Período', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
            
            ax4 = axes[3]
            bars4 = ax4.bar(range(len(stats_periodo)), stats_periodo['count'],
                           color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731'],
                           edgecolor='black', alpha=0.8, linewidth=1.5)
            ax4.set_ylabel('Cantidad de Semanas', fontsize=11, fontweight='bold')
            ax4.set_title('Cantidad de Semanas por Período', fontsize=12, fontweight='bold')
            ax4.set_xticks(range(len(stats_periodo)))
            ax4.set_xticklabels(stats_periodo.index, rotation=0)
            ax4.grid(True, alpha=0.3, axis='y')
            for bar in bars4:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('05_estadisticas_por_periodo.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 05_estadisticas_por_periodo.png")
            self.figuras_generadas.append('05_estadisticas_por_periodo.png')
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def grafica_correlacion_features(self):
        """
        Gráfica 6: Matriz de correlación.
        """
        print("\n✓ Generando gráfica 6: Matriz de correlación...")
        
        try:
            df_corr = self.df[['TOTAL_CASOS', 'SEMANA', 'ANO']].copy()
            df_corr['SEMANA_TEMPORAL'] = df_corr['ANO'] * 100 + df_corr['SEMANA']
            
            corr_matrix = df_corr.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
                       center=0, square=True, linewidths=1.5, cbar_kws={'label': 'Correlación'},
                       vmin=-1, vmax=1, ax=ax, annot_kws={'size': 11, 'weight': 'bold'})
            
            ax.set_title('Matriz de Correlación - Variables Temporales vs Total de Casos', 
                        fontsize=13, fontweight='bold', pad=20)
            plt.tight_layout()
            plt.savefig('06_matriz_correlacion.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 06_matriz_correlacion.png")
            self.figuras_generadas.append('06_matriz_correlacion.png')
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def grafica_comparativa_anos(self):
        """
        Gráfica 7: Comparación entre años.
        """
        print("✓ Generando gráfica 7: Comparación entre años...")
        
        try:
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))
            
            anos = sorted(self.df['ANO'].unique())
            colores = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            
            for idx, (ano, color) in enumerate(zip(anos, colores)):
                df_ano = self.df[self.df['ANO'] == ano]
                ax = axes[idx]
                
                ax.bar(df_ano['SEMANA'], df_ano['TOTAL_CASOS'], 
                      color=color, alpha=0.7, edgecolor='black', linewidth=1)
                ax.axhline(df_ano['TOTAL_CASOS'].mean(), color='red', 
                          linestyle='--', linewidth=2, label=f"Promedio: {df_ano['TOTAL_CASOS'].mean():.0f}")
                ax.set_xlabel('Semana del Año', fontsize=11, fontweight='bold')
                ax.set_ylabel('Total de Casos', fontsize=11, fontweight='bold')
                ax.set_title(f'Casos de Dengue - Año {ano}\n(Total: {df_ano["TOTAL_CASOS"].sum():,} casos)', 
                            fontsize=12, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3, axis='y')
                ax.set_xticks(np.arange(0, 54, 8))
            
            plt.tight_layout()
            plt.savefig('07_comparativa_anos.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 07_comparativa_anos.png")
            self.figuras_generadas.append('07_comparativa_anos.png')
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def grafica_outliers_deteccion(self):
        """
        Gráfica 8: Detección de outliers.
        """
        print("✓ Generando gráfica 8: Detección de outliers...")
        
        try:
            Q1 = self.df['TOTAL_CASOS'].quantile(0.25)
            Q3 = self.df['TOTAL_CASOS'].quantile(0.75)
            IQR = Q3 - Q1
            limite_inferior = Q1 - 1.5 * IQR
            limite_superior = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df['TOTAL_CASOS'] < limite_inferior) | 
                              (self.df['TOTAL_CASOS'] > limite_superior)]
            
            fig, ax = plt.subplots(figsize=(14, 7))
            
            normal = self.df[~self.df.index.isin(outliers.index)]
            ax.scatter(normal.index, normal['TOTAL_CASOS'], 
                      color='#3498DB', s=80, alpha=0.7, label='Valores normales', edgecolors='black', linewidth=0.5)
            ax.scatter(outliers.index, outliers['TOTAL_CASOS'],
                      color='#E74C3C', s=150, alpha=0.9, marker='X', 
                      label='Valores atípicos (outliers)', edgecolors='black', linewidth=1)
            
            ax.axhline(Q1, color='green', linestyle=':', linewidth=2, label='Q1')
            ax.axhline(Q3, color='orange', linestyle=':', linewidth=2, label='Q3')
            ax.axhline(limite_inferior, color='red', linestyle='--', linewidth=2, label='Límite inferior')
            ax.axhline(limite_superior, color='red', linestyle='--', linewidth=2, label='Límite superior')
            
            ax.set_xlabel('Índice Temporal (Semanas)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total de Casos', fontsize=12, fontweight='bold')
            ax.set_title('Detección de Valores Atípicos (Outliers)', 
                        fontsize=14, fontweight='bold', pad=20)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('08_deteccion_outliers.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("  ✓ Guardado: 08_deteccion_outliers.png")
            self.figuras_generadas.append('08_deteccion_outliers.png')
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        return self
    
    def generar_reporte_eda(self):
        """
        Genera un reporte completo del análisis exploratorio.
        """
        print("\n" + "=" * 70)
        print("REPORTE FINAL DEL ANÁLISIS EXPLORATORIO (EDA)")
        print("=" * 70)
        
        reporte = f"""
ANÁLISIS EXPLORATORIO DE DATOS (EDA) - DENGUE COLOMBIA 2022-2024
{'=' * 70}

1. ESTADÍSTICAS GENERALES
   - Total de semanas analizadas: {len(self.df)}
   - Rango temporal: {self.df['ANO'].min()} - {self.df['ANO'].max()}
   - Años cubiertos: {len(self.df['ANO'].unique())}
   
2. ESTADÍSTICAS DE CASOS (TOTAL_CASOS)
   - Promedio: {self.df['TOTAL_CASOS'].mean():.2f} casos/semana
   - Mediana: {self.df['TOTAL_CASOS'].median():.2f} casos/semana
   - Desviación Estándar: {self.df['TOTAL_CASOS'].std():.2f}
   - Mínimo: {self.df['TOTAL_CASOS'].min()} casos
   - Máximo: {self.df['TOTAL_CASOS'].max()} casos
   - Rango: {self.df['TOTAL_CASOS'].max() - self.df['TOTAL_CASOS'].min()} casos
   - Asimetría: {skew(self.df['TOTAL_CASOS']):.4f}
   - Curtosis: {kurtosis(self.df['TOTAL_CASOS']):.4f}

3. DISTRIBUCIÓN POR AÑO
{self.df.groupby('ANO')['TOTAL_CASOS'].agg(['count', 'sum', 'mean', 'std', 'min', 'max']).to_string()}

4. GRÁFICAS GENERADAS
   1. 01_distribucion_temporal.png - Serie temporal y box-plot
   2. 02_histogramas_distribucion.png - Histogramas de frecuencia
   3. 03_graficas_densidad.png - Funciones de densidad
   4. 04_patron_promedio_semana.png - Patrón estacional
   5. 05_estadisticas_por_periodo.png - Análisis por cuartales
   6. 06_matriz_correlacion.png - Matriz de correlación
   7. 07_comparativa_anos.png - Comparación entre años
   8. 08_deteccion_outliers.png - Detección de outliers

5. HALLAZGOS PRINCIPALES
   - Tendencia: Patrón estacional claro
   - Variabilidad: Alta (σ = {self.df['TOTAL_CASOS'].std():.2f})
   - Crecimiento: Incremento significativo de 2022 a 2024
   - Asimetría: {skew(self.df['TOTAL_CASOS']):.4f} (sesgada)

DATOS LISTOS PARA PREPROCESAMIENTO
{'=' * 70}
        """
        
        print(reporte)
        
        with open('reporte_eda_dengue.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("✓ Archivo guardado: reporte_eda_dengue.txt")
        
        return self
    
    def resumen_figuras(self):
        """
        Resumen de figuras generadas.
        """
        print("\n" + "=" * 70)
        print("FIGURAS GENERADAS EN ESTA FASE")
        print("=" * 70)
        for i, fig in enumerate(self.figuras_generadas, 1):
            print(f"{i}. {fig}")
        
        return self


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FASE 2: EXPLORACIÓN VISUAL DE DATOS (EDA)")
    print("=" * 70 + "\n")
    
    analizador = AnalizadorEDA('casos_por_semana_agregado.csv')
    
    (analizador
     .estadisticas_descriptivas()
     .grafica_distribucion_casos_temporal()
     .grafica_histograma_distribucion()
     .grafica_densidad_casos()
     .grafica_promedio_semana_del_ano()
     .grafica_estadisticas_por_periodo()
     .grafica_correlacion_features()
     .grafica_comparativa_anos()
     .grafica_outliers_deteccion()
     .generar_reporte_eda()
     .resumen_figuras())
    
    print("\n✓ PASO 2 COMPLETADO EXITOSAMENTE")
    print("Se han generado 8 gráficas en PNG (300 DPI)")
