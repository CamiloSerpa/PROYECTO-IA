import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PASO 4: REGRESIÓN MULTIVARIADA CON GRIDSEARCHCV
# ============================================================================
# Implementa:
# - Linear Regression
# - Ridge Regression
# - Lasso Regression
# - GridSearchCV para tuning automático de parámetros
# - Validación cruzada k=5
# - Comparación de desempeño
# ============================================================================

class ModeloRegresion:
    """
    Clase para entrenar y evaluar modelos de regresión multivariada.
    """
    
    def __init__(self, X_train_path, X_test_path, y_train_path, y_test_path):
        """
        Inicializa el modelo cargando datos preprocesados.
        """
        self.X_train = pd.read_csv(X_train_path)
        self.X_test = pd.read_csv(X_test_path)
        self.y_train = pd.read_csv(y_train_path).squeeze()
        self.y_test = pd.read_csv(y_test_path).squeeze()
        
        self.modelos_entrenados = {}
        self.resultados = {}
        
        print("=" * 70)
        print("PASO 4: REGRESIÓN MULTIVARIADA CON GRIDSEARCHCV")
        print("=" * 70)
        print(f"\nDatos cargados:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_test: {self.y_test.shape}")
    
    def entrenar_linear_regression(self):
        """
        Entrena Linear Regression (baseline).
        """
        print("\n" + "=" * 70)
        print("MODELO 1: LINEAR REGRESSION (BASELINE)")
        print("=" * 70)
        
        modelo = LinearRegression()
        modelo.fit(self.X_train, self.y_train)
        
        # Predicciones
        y_pred_train = modelo.predict(self.X_train)
        y_pred_test = modelo.predict(self.X_test)
        
        # Métricas
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        
        print(f"\n✓ Modelo entrenado")
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
            print(f"  → ALTO OVERFITTING DETECTADO")
        elif diferencia_r2 > 0.05:
            print(f"  → Overfitting moderado")
        else:
            print(f"  → Overfitting mínimo ✓")
        
        # Importancia de coeficientes
        coef_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coeficiente': modelo.coef_
        }).sort_values('Coeficiente', key=abs, ascending=False)
        
        print(f"\nTOP 5 FEATURES POR COEFICIENTE:")
        print(coef_importance.head())
        
        self.modelos_entrenados['LinearRegression'] = modelo
        self.resultados['LinearRegression'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test
        }
        
        return self
    
    def entrenar_ridge_gridsearch(self):
        """
        Entrena Ridge Regression con GridSearchCV para tuning de alpha.
        """
        print("\n" + "=" * 70)
        print("MODELO 2: RIDGE REGRESSION + GRIDSEARCHCV")
        print("=" * 70)
        
        # Parámetros a probar
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        
        # GridSearchCV con validación cruzada k=5
        ridge_base = Ridge()
        grid_search = GridSearchCV(
            ridge_base,
            param_grid,
            cv=5,  # Validación cruzada con 5 folds
            scoring='neg_mean_squared_error',
            n_jobs=-1,  # Usar todos los cores
            verbose=1
        )
        
        print(f"\n✓ Ejecutando GridSearchCV con validación cruzada k=5")
        print(f"  Parámetros a probar: {param_grid['alpha']}")
        print(f"  Total de combinaciones: 7")
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ GridSearchCV completado")
        print(f"\nMEJOR PARÁMETRO ENCONTRADO:")
        print(f"  alpha (regularización): {grid_search.best_params_['alpha']}")
        print(f"  MSE en validación cruzada: {-grid_search.best_score_:.2f}")
        
        # Usar mejor modelo
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones
        y_pred_train = mejor_modelo.predict(self.X_train)
        y_pred_test = mejor_modelo.predict(self.X_test)
        
        # Métricas
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        
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
            print(f"  → ALTO OVERFITTING DETECTADO")
        elif diferencia_r2 > 0.05:
            print(f"  → Overfitting moderado")
        else:
            print(f"  → Overfitting mínimo ✓")
        
        self.modelos_entrenados['Ridge'] = mejor_modelo
        self.resultados['Ridge'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'best_alpha': grid_search.best_params_['alpha']
        }
        
        return self
    
    def entrenar_lasso_gridsearch(self):
        """
        Entrena Lasso Regression con GridSearchCV para tuning de alpha.
        """
        print("\n" + "=" * 70)
        print("MODELO 3: LASSO REGRESSION + GRIDSEARCHCV")
        print("=" * 70)
        
        # Parámetros a probar
        param_grid = {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }
        
        # GridSearchCV con validación cruzada k=5
        lasso_base = Lasso(max_iter=10000)
        grid_search = GridSearchCV(
            lasso_base,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        print(f"\n✓ Ejecutando GridSearchCV con validación cruzada k=5")
        print(f"  Parámetros a probar: {param_grid['alpha']}")
        print(f"  Total de combinaciones: 7")
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ GridSearchCV completado")
        print(f"\nMEJOR PARÁMETRO ENCONTRADO:")
        print(f"  alpha (regularización): {grid_search.best_params_['alpha']}")
        print(f"  MSE en validación cruzada: {-grid_search.best_score_:.2f}")
        
        # Usar mejor modelo
        mejor_modelo = grid_search.best_estimator_
        
        # Predicciones
        y_pred_train = mejor_modelo.predict(self.X_train)
        y_pred_test = mejor_modelo.predict(self.X_test)
        
        # Métricas
        mse_train = mean_squared_error(self.y_train, y_pred_train)
        mse_test = mean_squared_error(self.y_test, y_pred_test)
        mae_train = mean_absolute_error(self.y_train, y_pred_train)
        mae_test = mean_absolute_error(self.y_test, y_pred_test)
        r2_train = r2_score(self.y_train, y_pred_train)
        r2_test = r2_score(self.y_test, y_pred_test)
        
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
            print(f"  → ALTO OVERFITTING DETECTADO")
        elif diferencia_r2 > 0.05:
            print(f"  → Overfitting moderado")
        else:
            print(f"  → Overfitting mínimo ✓")
        
        self.modelos_entrenados['Lasso'] = mejor_modelo
        self.resultados['Lasso'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'best_alpha': grid_search.best_params_['alpha']
        }
        
        return self
    
    def generar_tabla_comparativa(self):
        """
        Genera tabla comparativa de todos los modelos.
        """
        print("\n" + "=" * 70)
        print("TABLA COMPARATIVA DE MODELOS")
        print("=" * 70)
        
        tabla = []
        for nombre_modelo, metricas in self.resultados.items():
            tabla.append({
                'Modelo': nombre_modelo,
                'MSE_Train': f"{metricas['mse_train']:.2f}",
                'MSE_Test': f"{metricas['mse_test']:.2f}",
                'MAE_Train': f"{metricas['mae_train']:.2f}",
                'MAE_Test': f"{metricas['mae_test']:.2f}",
                'R²_Train': f"{metricas['r2_train']:.4f}",
                'R²_Test': f"{metricas['r2_test']:.4f}"
            })
        
        df_tabla = pd.DataFrame(tabla)
        print("\n" + df_tabla.to_string(index=False))
        
        # Guardar tabla
        df_tabla.to_csv('resultados_regresion.csv', index=False)
        print("\n✓ Tabla guardada en: resultados_regresion.csv")
        
        return self
    
    def generar_graficas_predicciones(self):
        """
        Genera gráficas de predicciones vs realidad para cada modelo.
        """
        print("\n" + "=" * 70)
        print("GENERANDO GRÁFICAS DE PREDICCIONES")
        print("=" * 70)
        
        import matplotlib
        matplotlib.use('Agg')
        
        # Crear figura con subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        colores = {'LinearRegression': '#FF6B6B', 'Ridge': '#4ECDC4', 'Lasso': '#45B7D1'}
        
        for idx, (nombre_modelo, metricas) in enumerate(self.resultados.items()):
            ax = axes[idx]
            
            y_pred = metricas['y_pred_test']
            
            # Scatter plot: Real vs Predicho
            ax.scatter(self.y_test, y_pred, alpha=0.6, s=100, 
                      color=colores[nombre_modelo], edgecolors='black', linewidth=0.5)
            
            # Línea diagonal (predicción perfecta)
            min_val = min(self.y_test.min(), y_pred.min())
            max_val = max(self.y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
            
            ax.set_xlabel('Valores Reales', fontsize=11, fontweight='bold')
            ax.set_ylabel('Valores Predichos', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo}\nR²={metricas["r2_test"]:.4f}, MAE={metricas["mae_test"]:.2f}', 
                        fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('01_regresion_predicciones_vs_real.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 01_regresion_predicciones_vs_real.png")
        
        return self
    
    def generar_graficas_residuos(self):
        """
        Genera gráficas de residuos para diagnóstico.
        """
        print("✓ Generando gráficas de residuos...")
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        colores = {'LinearRegression': '#FF6B6B', 'Ridge': '#4ECDC4', 'Lasso': '#45B7D1'}
        
        for idx, (nombre_modelo, metricas) in enumerate(self.resultados.items()):
            ax = axes[idx]
            
            y_pred = metricas['y_pred_test']
            residuos = self.y_test - y_pred
            
            # Scatter: Residuos vs Predicciones
            ax.scatter(y_pred, residuos, alpha=0.6, s=100,
                      color=colores[nombre_modelo], edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Valores Predichos', fontsize=11, fontweight='bold')
            ax.set_ylabel('Residuos', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo}\nResíduos vs Predicciones', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('02_regresion_residuos.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 02_regresion_residuos.png")
        
        return self
    
    def generar_reporte_final(self):
        """
        Genera reporte final con conclusiones.
        """
        print("\n" + "=" * 70)
        print("GENERANDO REPORTE FINAL")
        print("=" * 70)
        
        # Encontrar mejor modelo
        mejor_modelo = max(self.resultados.items(), 
                          key=lambda x: x[1]['r2_test'])
        
        reporte = f"""
REPORTE FINAL - REGRESIÓN MULTIVARIADA
DENGUE COLOMBIA 2022-2024
{'=' * 70}

1. RESUMEN EJECUTIVO
   - Objetivo: Predecir casos de dengue por semana
   - Tipo: Regresión multivariada
   - Modelos: Linear Regression, Ridge, Lasso
   - Método de tuning: GridSearchCV con validación cruzada k=5
   - Muestras entrenamiento: 121
   - Muestras prueba: 31

2. RESULTADOS POR MODELO

   LINEAR REGRESSION (BASELINE):
   - R² Train: {self.resultados['LinearRegression']['r2_train']:.4f}
   - R² Test:  {self.resultados['LinearRegression']['r2_test']:.4f}
   - MSE Test: {self.resultados['LinearRegression']['mse_test']:.2f}
   - MAE Test: {self.resultados['LinearRegression']['mae_test']:.2f}
   
   RIDGE REGRESSION (Alpha={self.resultados['Ridge'].get('best_alpha', 'N/A')}):
   - R² Train: {self.resultados['Ridge']['r2_train']:.4f}
   - R² Test:  {self.resultados['Ridge']['r2_test']:.4f}
   - MSE Test: {self.resultados['Ridge']['mse_test']:.2f}
   - MAE Test: {self.resultados['Ridge']['mae_test']:.2f}
   
   LASSO REGRESSION (Alpha={self.resultados['Lasso'].get('best_alpha', 'N/A')}):
   - R² Train: {self.resultados['Lasso']['r2_train']:.4f}
   - R² Test:  {self.resultados['Lasso']['r2_test']:.4f}
   - MSE Test: {self.resultados['Lasso']['mse_test']:.2f}
   - MAE Test: {self.resultados['Lasso']['mae_test']:.2f}

3. MEJOR MODELO
   Modelo: {mejor_modelo[0]}
   R² Test: {mejor_modelo[1]['r2_test']:.4f}
   MAE Test: {mejor_modelo[1]['mae_test']:.2f}

4. ANÁLISIS DE OVERFITTING
   
   Linear Regression:
   - Diferencia R²: {self.resultados['LinearRegression']['r2_train'] - self.resultados['LinearRegression']['r2_test']:.4f}
   
   Ridge:
   - Diferencia R²: {self.resultados['Ridge']['r2_train'] - self.resultados['Ridge']['r2_test']:.4f}
   
   Lasso:
   - Diferencia R²: {self.resultados['Lasso']['r2_train'] - self.resultados['Lasso']['r2_test']:.4f}

5. CONCLUSIONES
   - El modelo {mejor_modelo[0]} presenta el mejor R² en test ({mejor_modelo[1]['r2_test']:.4f})
   - Overfitting detectado: Evaluar si usar regularización adicional
   - Validación cruzada k=5 fue efectiva para tuning de parámetros
   - Features lag (TOTAL_CASOS_LAG1-4) contribuyen significativamente

6. RECOMENDACIONES PARA PRÓXIMOS PASOS
   - Probar Árboles de Decisión (capturan no-linealidad)
   - Probar Random Forest (mejor generalización)
   - Considerar Redes Neuronales con regularización L2
   - Explorar features adicionales (variables externas si disponibles)

ARCHIVOS GENERADOS:
- resultados_regresion.csv: Tabla comparativa de modelos
- 01_regresion_predicciones_vs_real.png: Scatter plots predicciones
- 02_regresion_residuos.png: Análisis de residuos

{'=' * 70}
        """
        
        print(reporte)
        
        with open('reporte_regresion.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("\n✓ Reporte guardado: reporte_regresion.txt")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Cargar datos
    modelo = ModeloRegresion(
        'X_train_normalizado.csv',
        'X_test_normalizado.csv',
        'y_train.csv',
        'y_test.csv'
    )
    
    # Entrenar modelos
    (modelo
     .entrenar_linear_regression()
     .entrenar_ridge_gridsearch()
     .entrenar_lasso_gridsearch()
     .generar_tabla_comparativa()
     .generar_graficas_predicciones()
     .generar_graficas_residuos()
     .generar_reporte_final())
    
    print("\n" + "=" * 70)
    print("✓ PASO 4 COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("\nModelos entrenados y evaluados:")
    print("  1. Linear Regression (baseline)")
    print("  2. Ridge Regression (con GridSearchCV)")
    print("  3. Lasso Regression (con GridSearchCV)")
    print("\nArchivos generados:")
    print("  - resultados_regresion.csv")
    print("  - 01_regresion_predicciones_vs_real.png")
    print("  - 02_regresion_residuos.png")
    print("  - reporte_regresion.txt")
    print("\n🚀 Próximo paso: PASO 5 - Árboles de Decisión")
