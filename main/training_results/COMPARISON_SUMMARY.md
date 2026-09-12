# 🏎️ NXP Cup — Consolidated Model Training & Evaluation Summary

**Generated on:** 2026-09-10 09:50:35  
**Directory:** `main/training_results`

---

## 📊 Benchmark Comparativo de Desempenho

| Modelo | Parâmetros | Época de Melhor Val | Val MAE (°) | Test MAE (°) | Test RMSE (°) | Test $R^2$ Score | Max Error (°) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **micro_vit** | 359,841 | 44 | 5.34° | **5.66°** | 10.06° | 0.648 | 54.02° |
| **palhano_cnn** | 886,037 | 45 | 6.36° | **6.43°** | 9.95° | 0.656 | 49.76° |
| **dave2** | 1,595,511 | 48 | 4.97° | **5.15°** | 9.12° | 0.711 | 52.10° |

---

## 📂 Estrutura de Arquivos Gerada

```text
main/training_results/
├── micro_vit/
│   ├── best_model.pt               # Pesos ótimos PyTorch
│   ├── evaluation_summary.json     # Métricas JSON
│   ├── learning_curves.png         # Evolução da Huber Loss & MAE
│   ├── steer_scatter.png           # Dispersão Predição vs Ground Truth
│   └── residual_distribution.png   # Histograma de resíduos
├── palhano_cnn/
│   └── [mesma estrutura]
├── dave2/
│   └── [mesma estrutura]
├── model_comparison_report.json    # Relatório JSON agregado
└── COMPARISON_SUMMARY.md           # Este relatório
```
