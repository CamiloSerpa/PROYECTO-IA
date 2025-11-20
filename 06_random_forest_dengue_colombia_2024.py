import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PASO 6: RANDOM FOREST CON GRIDSEARCHCV
# ============================================================================
# Implementa:
# - Random Forest Regressor (múltiples árboles en paralelo)
# - GridSearchCV para tuning de n_estimators, max_depth, min_samples_leaf
# - Validación cruzada k=5
# - Feature importance (promedio de importancia en todos los árboles)
# - Out-of-Bag (OOB) score evaluation
# - Comparación con Regresión y Árboles individuales
# ============================================================================

class ModeloRandomForest:
    """
    Clase para entrenar y evaluar Random Forest.
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
        print("PASO 6: RANDOM FOREST CON GRIDSEARCHCV")
        print("=" * 70)
        print(f"\nDatos cargados:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_test: {self.y_test.shape}")
    
    def entrenar_random_forest_baseline(self):
        """
        Entrena Random Forest con parámetros por defecto (baseline).
        ✅ CORRECCIÓN: oob_score=True para poder acceder a oob_score_
        """
        print("\n" + "=" * 70)
        print("MODELO 1: RANDOM FOREST - BASELINE (100 árboles)")
        print("=" * 70)
        
        # ✅ CORRECCIÓN: Agregué oob_score=True
        modelo = RandomForestRegressor(
            n_estimators=100,
            oob_score=True,  # ✅ CRÍTICO para obtener oob_score_
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
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
        
        # OOB Score (Out-of-Bag)
        # Random Forest entrena cada árbol con ~63% de datos (bootstrap)
        # Usa el ~37% restante como validación automática
        oob_score = modelo.oob_score_
        
        print(f"\n✓ Modelo entrenado")
        print(f"  Número de árboles: 100")
        print(f"  OOB Score (validación automática): {oob_score:.4f}")
        
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
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importancia': modelo.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        print(f"\nTOP 5 FEATURES POR IMPORTANCIA:")
        print(feature_importance.head())
        
        self.modelos_entrenados['RF_Baseline'] = modelo
        self.resultados['RF_Baseline'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'oob_score': oob_score,
            'n_estimators': 100,
            'feature_importance': feature_importance
        }
        
        return self
    
    def entrenar_random_forest_gridsearch(self):
        """
        Entrena Random Forest con GridSearchCV para tuning óptimo.
        """
        print("\n" + "=" * 70)
        print("MODELO 2: RANDOM FOREST + GRIDSEARCHCV")
        print("=" * 70)
        
        # Parámetros a probar
        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # GridSearchCV con validación cruzada k=5
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            rf_base,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0
        )
        
        print(f"\n✓ Ejecutando GridSearchCV con validación cruzada k=5")
        print(f"  Combinaciones de parámetros: 4 × 4 × 2 × 2 = 64")
        print(f"  Total de fits: 64 × 5 folds = 320")
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ GridSearchCV completado")
        print(f"\nMEJORES PARÁMETROS ENCONTRADOS:")
        print(f"  n_estimators: {grid_search.best_params_['n_estimators']}")
        print(f"  max_depth: {grid_search.best_params_['max_depth']}")
        print(f"  min_samples_split: {grid_search.best_params_['min_samples_split']}")
        print(f"  min_samples_leaf: {grid_search.best_params_['min_samples_leaf']}")
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
            print(f"  → ALTO OVERFITTING DETECTADO ❌")
        elif diferencia_r2 > 0.05:
            print(f"  → Overfitting moderado")
        else:
            print(f"  → Overfitting mínimo ✓")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importancia': mejor_modelo.feature_importances_
        }).sort_values('Importancia', ascending=False)
        
        print(f"\nTOP 5 FEATURES POR IMPORTANCIA:")
        print(feature_importance.head())
        
        self.modelos_entrenados['RF_GridSearchCV'] = mejor_modelo
        self.resultados['RF_GridSearchCV'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'n_estimators': grid_search.best_params_['n_estimators'],
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def generar_tabla_comparativa(self):
        """
        Genera tabla comparativa de Random Forest.
        """
        print("\n" + "=" * 70)
        print("TABLA COMPARATIVA DE RANDOM FOREST")
        print("=" * 70)
        
        tabla = []
        for nombre_modelo, metricas in self.resultados.items():
            tabla.append({
                'Modelo': nombre_modelo,
                'N_Estimators': metricas.get('n_estimators', '-'),
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
        df_tabla.to_csv('resultados_random_forest.csv', index=False)
        print("\n✓ Tabla guardada en: resultados_random_forest.csv")
        
        return self
    
    def generar_graficas_predicciones(self):
        """
        Genera gráficas de predicciones vs realidad.
        """
        print("\n" + "=" * 70)
        print("GENERANDO GRÁFICAS DE PREDICCIONES")
        print("=" * 70)
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colores = {'RF_Baseline': '#FF6B6B', 'RF_GridSearchCV': '#4ECDC4'}
        
        for idx, (nombre_modelo, metricas) in enumerate(self.resultados.items()):
            ax = axes[idx]
            
            y_pred = metricas['y_pred_test']
            
            # Scatter plot
            ax.scatter(self.y_test, y_pred, alpha=0.6, s=100, 
                      color=colores[nombre_modelo], edgecolors='black', linewidth=0.5)
            
            # Línea diagonal
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
        plt.savefig('01_random_forest_predicciones_vs_real.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 01_random_forest_predicciones_vs_real.png")
        
        return self
    
    def generar_graficas_feature_importance(self):
        """
        Genera gráficas de importancia de features.
        """
        print("✓ Generando gráficas de feature importance...")
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        colores_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#F7B731', '#5F27CD']
        
        for idx, (nombre_modelo, metricas) in enumerate(self.resultados.items()):
            ax = axes[idx]
            
            feature_imp = metricas['feature_importance'].head(10)
            
            bars = ax.barh(feature_imp['Feature'], feature_imp['Importancia'],
                           color=colores_bar, edgecolor='black', alpha=0.8, linewidth=1)
            
            ax.set_xlabel('Importancia', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo}\nTop 10 Features', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('02_random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 02_random_forest_feature_importance.png")
        
        return self
    
    def generar_graficas_residuos(self):
        """
        Genera gráficas de residuos.
        """
        print("✓ Generando gráficas de residuos...")
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colores = {'RF_Baseline': '#FF6B6B', 'RF_GridSearchCV': '#4ECDC4'}
        
        for idx, (nombre_modelo, metricas) in enumerate(self.resultados.items()):
            ax = axes[idx]
            
            y_pred = metricas['y_pred_test']
            residuos = self.y_test - y_pred
            
            ax.scatter(y_pred, residuos, alpha=0.6, s=100,
                      color=colores[nombre_modelo], edgecolors='black', linewidth=0.5)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            
            ax.set_xlabel('Valores Predichos', fontsize=11, fontweight='bold')
            ax.set_ylabel('Residuos', fontsize=11, fontweight='bold')
            ax.set_title(f'{nombre_modelo}\nResíduos vs Predicciones', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('03_random_forest_residuos.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 03_random_forest_residuos.png")
        
        return self
    
    def generar_comparativa_total(self):
        """
        Compara Random Forest vs TODOS los modelos previos.
        """
        print("\n" + "=" * 70)
        print("COMPARACIÓN TOTAL: RANDOM FOREST vs TODOS LOS MODELOS")
        print("=" * 70)
        
        try:
            df_regresion = pd.read_csv('resultados_regresion.csv')
            df_arboles = pd.read_csv('resultados_arboles.csv')
            df_rf = pd.read_csv('resultados_random_forest.csv')
            
            print("\n" + "=" * 70)
            print("PASO 4 - REGRESIÓN MULTIVARIADA")
            print("=" * 70)
            print(df_regresion.to_string(index=False))
            
            print("\n" + "=" * 70)
            print("PASO 5 - ÁRBOLES DE DECISIÓN")
            print("=" * 70)
            print(df_arboles.to_string(index=False))
            
            print("\n" + "=" * 70)
            print("PASO 6 - RANDOM FOREST")
            print("=" * 70)
            print(df_rf.to_string(index=False))
            
            print("\n" + "=" * 70)
            print("RANKING FINAL (por R² en TEST)")
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
            
            ranking_data.sort(key=lambda x: x[1], reverse=True)
            
            print("\n")
            for i, (modelo, r2) in enumerate(ranking_data, 1):
                print(f"{i:2d}. {modelo:<35s} R² = {r2:.4f}")
            
            print("\n" + "=" * 70)
            print("CONCLUSIÓN")
            print("=" * 70)
            print(f"\nMejor modelo GLOBAL: {ranking_data[0][0]} (R² = {ranking_data[0][1]:.4f})")
        
        except FileNotFoundError as e:
            print(f"\n⚠️  No se encontraron archivos de pasos anteriores")
            print("  Asegúrate de haber ejecutado PASO 4 y PASO 5 primero")
        
        return self
    
    def generar_reporte_final(self):
        """
        Genera reporte final.
        """
        print("\n" + "=" * 70)
        print("GENERANDO REPORTE FINAL")
        print("=" * 70)
        
        mejor_modelo_rf = max(self.resultados.items(), 
                              key=lambda x: x[1]['r2_test'])
        
        reporte = f"""
REPORTE FINAL - RANDOM FOREST
DENGUE COLOMBIA 2022-2024
{'=' * 70}

1. RESUMEN EJECUTIVO
   - Objetivo: Predecir casos de dengue usando ensambles de árboles
   - Modelos: Random Forest Baseline + Random Forest con GridSearchCV
   - Muestras entrenamiento: 121
   - Muestras prueba: 31

2. RESULTADOS POR MODELO

   RANDOM FOREST - BASELINE (100 árboles):
   - OOB Score: {self.resultados['RF_Baseline'].get('oob_score', 'N/A'):.4f}
   - R² Train: {self.resultados['RF_Baseline']['r2_train']:.4f}
   - R² Test:  {self.resultados['RF_Baseline']['r2_test']:.4f}
   - MSE Test: {self.resultados['RF_Baseline']['mse_test']:.2f}
   - MAE Test: {self.resultados['RF_Baseline']['mae_test']:.2f}
   
   RANDOM FOREST + GRIDSEARCHCV:
   - N_Estimators: {self.resultados['RF_GridSearchCV']['n_estimators']}
   - R² Train: {self.resultados['RF_GridSearchCV']['r2_train']:.4f}
   - R² Test:  {self.resultados['RF_GridSearchCV']['r2_test']:.4f}
   - MSE Test: {self.resultados['RF_GridSearchCV']['mse_test']:.2f}
   - MAE Test: {self.resultados['RF_GridSearchCV']['mae_test']:.2f}

3. MEJOR MODELO RANDOM FOREST
   Modelo: {mejor_modelo_rf[0]}
   R² Test: {mejor_modelo_rf[1]['r2_test']:.4f}
   MAE Test: {mejor_modelo_rf[1]['mae_test']:.2f}

4. ANÁLISIS DE OVERFITTING
   
   Random Forest (Baseline):
   - Diferencia R²: {self.resultados['RF_Baseline']['r2_train'] - self.resultados['RF_Baseline']['r2_test']:.4f}
   
   Random Forest (GridSearchCV):
   - Diferencia R²: {self.resultados['RF_GridSearchCV']['r2_train'] - self.resultados['RF_GridSearchCV']['r2_test']:.4f}

5. FEATURE IMPORTANCE (Top 3)
   
   {self.resultados['RF_GridSearchCV']['feature_importance'].head(3).to_string()}

6. VENTAJAS DE RANDOM FOREST
   
   ✅ Múltiples árboles = reducción de overfitting
   ✅ Robustez a outliers
   ✅ Feature importance estable
   ✅ Paralelizable (n_jobs=-1)
   ✅ OOB Score para validación automática

7. PRÓXIMO PASO: REDES NEURONALES (PASO 7)
   
   Se espera:
   - R² similar (~0.975-0.99)
   - Posibles mejoras en MAE
   - Mayor capacidad para capturar no-linealidades

{'=' * 70}
        """
        
        print(reporte)
        
        with open('reporte_random_forest.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("\n✓ Reporte guardado: reporte_random_forest.txt")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    modelo = ModeloRandomForest(
        'X_train_normalizado.csv',
        'X_test_normalizado.csv',
        'y_train.csv',
        'y_test.csv'
    )
    
    (modelo
     .entrenar_random_forest_baseline()
     .entrenar_random_forest_gridsearch()
     .generar_tabla_comparativa()
     .generar_graficas_predicciones()
     .generar_graficas_feature_importance()
     .generar_graficas_residuos()
     .generar_comparativa_total()
     .generar_reporte_final())
    
    print("\n" + "=" * 70)
    print("✓ PASO 6 COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("\nModelos entrenados:")
    print("  1. Random Forest (baseline - 100 árboles)")
    print("  2. Random Forest (con GridSearchCV)")
    print("\nArchivos generados:")
    print("  - resultados_random_forest.csv")
    print("  - 01_random_forest_predicciones_vs_real.png")
    print("  - 02_random_forest_feature_importance.png")
    print("  - 03_random_forest_residuos.png")
    print("  - reporte_random_forest.txt")
    print("\n🚀 Próximo paso: PASO 7 - Redes Neuronales (MLP + DNN)")
