import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PASO 5: ÁRBOLES DE DECISIÓN CON GRIDSEARCHCV
# ============================================================================
# Implementa:
# - Decision Tree Regressor con diferentes profundidades
# - GridSearchCV para tuning automático
# - Validación cruzada k=5
# - Feature importance (qué features usa el árbol)
# - Comparación con modelos de Regresión (PASO 4)
# ============================================================================

class ModeloArboles:
    """
    Clase para entrenar y evaluar árboles de decisión.
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
        print("PASO 5: ÁRBOLES DE DECISIÓN CON GRIDSEARCHCV")
        print("=" * 70)
        print(f"\nDatos cargados:")
        print(f"  X_train: {self.X_train.shape}")
        print(f"  X_test: {self.X_test.shape}")
        print(f"  y_train: {self.y_train.shape}")
        print(f"  y_test: {self.y_test.shape}")
    
    def entrenar_decision_tree_profundo(self):
        """
        Entrena árbol de decisión con máxima profundidad (sin restricción).
        """
        print("\n" + "=" * 70)
        print("MODELO 1: DECISION TREE - SIN RESTRICCIÓN (max_depth=None)")
        print("=" * 70)
        
        modelo = DecisionTreeRegressor(random_state=42)
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
        print(f"  Profundidad del árbol: {modelo.get_depth()}")
        print(f"  Número de hojas: {modelo.get_n_leaves()}")
        
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
        
        self.modelos_entrenados['DT_Profundo'] = modelo
        self.resultados['DT_Profundo'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'profundidad': modelo.get_depth(),
            'hojas': modelo.get_n_leaves(),
            'feature_importance': feature_importance
        }
        
        return self
    
    def entrenar_decision_tree_gridsearch(self):
        """
        Entrena árbol de decisión con GridSearchCV para tuning de profundidad.
        """
        print("\n" + "=" * 70)
        print("MODELO 2: DECISION TREE + GRIDSEARCHCV")
        print("=" * 70)
        
        # Parámetros a probar
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # GridSearchCV con validación cruzada k=5
        dt_base = DecisionTreeRegressor(random_state=42)
        grid_search = GridSearchCV(
            dt_base,
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=0  # Sin verbosidad para evitar spam
        )
        
        print(f"\n✓ Ejecutando GridSearchCV con validación cruzada k=5")
        print(f"  Combinaciones de parámetros: 7 × 3 × 3 = 63")
        print(f"  Total de fits: 63 × 5 folds = 315")
        
        grid_search.fit(self.X_train, self.y_train)
        
        print(f"\n✓ GridSearchCV completado")
        print(f"\nMEJORES PARÁMETROS ENCONTRADOS:")
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
        
        self.modelos_entrenados['DT_GridSearchCV'] = mejor_modelo
        self.resultados['DT_GridSearchCV'] = {
            'mse_train': mse_train,
            'mse_test': mse_test,
            'mae_train': mae_train,
            'mae_test': mae_test,
            'r2_train': r2_train,
            'r2_test': r2_test,
            'y_pred_test': y_pred_test,
            'profundidad': mejor_modelo.get_depth(),
            'hojas': mejor_modelo.get_n_leaves(),
            'feature_importance': feature_importance,
            'best_params': grid_search.best_params_
        }
        
        return self
    
    def generar_tabla_comparativa(self):
        """
        Genera tabla comparativa de todos los modelos de árboles.
        """
        print("\n" + "=" * 70)
        print("TABLA COMPARATIVA DE ÁRBOLES")
        print("=" * 70)
        
        tabla = []
        for nombre_modelo, metricas in self.resultados.items():
            tabla.append({
                'Modelo': nombre_modelo,
                'Profundidad': metricas['profundidad'],
                'Hojas': metricas['hojas'],
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
        df_tabla.to_csv('resultados_arboles.csv', index=False)
        print("\n✓ Tabla guardada en: resultados_arboles.csv")
        
        return self
    
    def generar_graficas_predicciones(self):
        """
        Genera gráficas de predicciones vs realidad para cada árbol.
        """
        print("\n" + "=" * 70)
        print("GENERANDO GRÁFICAS DE PREDICCIONES")
        print("=" * 70)
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colores = {'DT_Profundo': '#FF6B6B', 'DT_GridSearchCV': '#4ECDC4'}
        
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
        plt.savefig('01_arboles_predicciones_vs_real.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 01_arboles_predicciones_vs_real.png")
        
        return self
    
    def generar_graficas_feature_importance(self):
        """
        Genera gráficas de importancia de features para cada árbol.
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
            ax.set_title(f'{nombre_modelo}\nTop 10 Features (Profundidad={metricas["profundidad"]})', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Anotar valores en barras
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2.,
                       f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('02_arboles_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 02_arboles_feature_importance.png")
        
        return self
    
    def generar_graficas_residuos(self):
        """
        Genera gráficas de residuos para diagnóstico.
        """
        print("✓ Generando gráficas de residuos...")
        
        import matplotlib
        matplotlib.use('Agg')
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        colores = {'DT_Profundo': '#FF6B6B', 'DT_GridSearchCV': '#4ECDC4'}
        
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
        plt.savefig('03_arboles_residuos.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("✓ Guardado: 03_arboles_residuos.png")
        
        return self
    
    def generar_comparativa_con_regresion(self):
        """
        Compara resultados de árboles vs regresión (PASO 4).
        """
        print("\n" + "=" * 70)
        print("COMPARACIÓN ÁRBOLES vs REGRESIÓN (PASO 4)")
        print("=" * 70)
        
        # Cargar resultados de regresión
        try:
            df_regresion = pd.read_csv('resultados_regresion.csv')
            
            print("\nRESULTADOS REGRESIÓN (PASO 4):")
            print(df_regresion.to_string(index=False))
            
            print("\nRESULTADOS ÁRBOLES (PASO 5):")
            df_arboles = pd.read_csv('resultados_arboles.csv')
            print(df_arboles.to_string(index=False))
            
            # Análisis comparativo
            print("\n" + "=" * 70)
            print("ANÁLISIS COMPARATIVO")
            print("=" * 70)
            
            r2_ridge = 0.9775  # Del PASO 4
            r2_mejor_arbol = float(df_arboles.iloc[df_arboles['R²_Test'].astype(float).argmax()]['R²_Test'])
            
            print(f"\nMejor R² Regresión (Ridge): 0.9775")
            print(f"Mejor R² Árboles: {r2_mejor_arbol:.4f}")
            
            if r2_mejor_arbol > r2_ridge:
                print(f"✅ ÁRBOLES GANAN por {(r2_mejor_arbol - r2_ridge)*100:.2f}%")
            elif r2_mejor_arbol < r2_ridge:
                print(f"❌ REGRESIÓN GANA por {(r2_ridge - r2_mejor_arbol)*100:.2f}%")
            else:
                print(f"⚖️  EMPATE en rendimiento")
        
        except FileNotFoundError:
            print("\n⚠️  No se encontró archivo resultados_regresion.csv")
            print("  (Asegúrate de haber ejecutado PASO 4 primero)")
        
        return self
    
    def generar_reporte_final(self):
        """
        Genera reporte final con conclusiones.
        """
        print("\n" + "=" * 70)
        print("GENERANDO REPORTE FINAL")
        print("=" * 70)
        
        # Encontrar mejor modelo
        mejor_modelo_arbol = max(self.resultados.items(), 
                                 key=lambda x: x[1]['r2_test'])
        
        reporte = f"""
REPORTE FINAL - ÁRBOLES DE DECISIÓN
DENGUE COLOMBIA 2022-2024
{'=' * 70}

1. RESUMEN EJECUTIVO
   - Objetivo: Predecir casos de dengue por semana usando árboles
   - Modelos: Decision Tree sin restricción + Decision Tree con GridSearchCV
   - Método de tuning: GridSearchCV con validación cruzada k=5
   - Muestras entrenamiento: 121
   - Muestras prueba: 31

2. RESULTADOS POR MODELO

   DECISION TREE - SIN RESTRICCIÓN:
   - Profundidad: {self.resultados['DT_Profundo']['profundidad']}
   - Número de hojas: {self.resultados['DT_Profundo']['hojas']}
   - R² Train: {self.resultados['DT_Profundo']['r2_train']:.4f}
   - R² Test:  {self.resultados['DT_Profundo']['r2_test']:.4f}
   - MSE Test: {self.resultados['DT_Profundo']['mse_test']:.2f}
   - MAE Test: {self.resultados['DT_Profundo']['mae_test']:.2f}
   
   DECISION TREE + GRIDSEARCHCV:
   - Profundidad: {self.resultados['DT_GridSearchCV']['profundidad']}
   - Número de hojas: {self.resultados['DT_GridSearchCV']['hojas']}
   - R² Train: {self.resultados['DT_GridSearchCV']['r2_train']:.4f}
   - R² Test:  {self.resultados['DT_GridSearchCV']['r2_test']:.4f}
   - MSE Test: {self.resultados['DT_GridSearchCV']['mse_test']:.2f}
   - MAE Test: {self.resultados['DT_GridSearchCV']['mae_test']:.2f}

3. MEJOR MODELO
   Modelo: {mejor_modelo_arbol[0]}
   R² Test: {mejor_modelo_arbol[1]['r2_test']:.4f}
   MAE Test: {mejor_modelo_arbol[1]['mae_test']:.2f}

4. ANÁLISIS DE OVERFITTING
   
   Decision Tree (Sin restricción):
   - Diferencia R²: {self.resultados['DT_Profundo']['r2_train'] - self.resultados['DT_Profundo']['r2_test']:.4f}
   - Interpretación: {'ALTO OVERFITTING' if (self.resultados['DT_Profundo']['r2_train'] - self.resultados['DT_Profundo']['r2_test']) > 0.15 else 'Overfitting moderado' if (self.resultados['DT_Profundo']['r2_train'] - self.resultados['DT_Profundo']['r2_test']) > 0.05 else 'Overfitting mínimo'}
   
   Decision Tree (GridSearchCV):
   - Diferencia R²: {self.resultados['DT_GridSearchCV']['r2_train'] - self.resultados['DT_GridSearchCV']['r2_test']:.4f}
   - Interpretación: {'ALTO OVERFITTING' if (self.resultados['DT_GridSearchCV']['r2_train'] - self.resultados['DT_GridSearchCV']['r2_test']) > 0.15 else 'Overfitting moderado' if (self.resultados['DT_GridSearchCV']['r2_train'] - self.resultados['DT_GridSearchCV']['r2_test']) > 0.05 else 'Overfitting mínimo'}

5. FEATURE IMPORTANCE
   
   Top 3 Features más importantes:
   {self.resultados['DT_GridSearchCV']['feature_importance'].head(3).to_string()}

6. CONCLUSIONES
   - Árboles vs Regresión: Comparar R² test (~0.975 esperado en regresión)
   - GridSearchCV mejoró la regularización automáticamente
   - Features lag siguen siendo importantes para árboles
   - Profundidad óptima: {self.resultados['DT_GridSearchCV']['profundidad']} (más somero que sin restricción)

7. RECOMENDACIONES PARA PRÓXIMOS PASOS
   - PASO 6: Probar Random Forest (múltiples árboles = mejor generalización)
   - PASO 7: Redes Neuronales pueden capturar interacciones complejas
   - Considerar ensambles (votación entre modelos)

ARCHIVOS GENERADOS:
- resultados_arboles.csv: Tabla comparativa de modelos
- 01_arboles_predicciones_vs_real.png: Scatter plots predicciones
- 02_arboles_feature_importance.png: Importancia de features
- 03_arboles_residuos.png: Análisis de residuos

{'=' * 70}
        """
        
        print(reporte)
        
        with open('reporte_arboles.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        
        print("\n✓ Reporte guardado: reporte_arboles.txt")


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # Cargar datos (usando datos normalizados como antes)
    modelo = ModeloArboles(
        'X_train_normalizado.csv',
        'X_test_normalizado.csv',
        'y_train.csv',
        'y_test.csv'
    )
    
    # Entrenar modelos
    (modelo
     .entrenar_decision_tree_profundo()
     .entrenar_decision_tree_gridsearch()
     .generar_tabla_comparativa()
     .generar_graficas_predicciones()
     .generar_graficas_feature_importance()
     .generar_graficas_residuos()
     .generar_comparativa_con_regresion()
     .generar_reporte_final())
    
    print("\n" + "=" * 70)
    print("✓ PASO 5 COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print("\nModelos entrenados y evaluados:")
    print("  1. Decision Tree (sin restricción)")
    print("  2. Decision Tree (con GridSearchCV)")
    print("\nArchivos generados:")
    print("  - resultados_arboles.csv")
    print("  - 01_arboles_predicciones_vs_real.png")
    print("  - 02_arboles_feature_importance.png")
    print("  - 03_arboles_residuos.png")
    print("  - reporte_arboles.txt")
    print("\n🚀 Próximo paso: PASO 6 - Random Forest")
