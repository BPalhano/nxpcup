# 🏎️ NXP Cup Autonomous Driving — MicroViT & Edge AI MLOps

Sistema de condução autônoma end-to-end de alta performance desenvolvido para a competição NXP Cup, superando a arquitetura NVIDIA DAVE-2 em precisão e latência através de visão computacional baseada em Transformers leves (**MicroViT**) e inferência otimizada via **TensorRT INT8**.

---

## 🎯 Arquitetura & Engenharia

* **Modelos:** Baseline NVIDIA DAVE-2, Palhano CNN (Paper) e MicroViT-NXP (Transformer Híbrido).
* **Otimização Edge:** Exportação ONNX Dinâmica com calibração INT8 Entropy (`IInt8EntropyCalibrator2`).
* **MLOps:** Rastreamento com MLflow, versionamento com DVC e detecção de Data Drift com Evidently AI.
* **Deploy:** API assíncrona FastAPI com Docker multi-stage.

---

## 🚀 Como Executar

```bash
# Instalar dependências
make install

# Executar testes
make test

# Executar linter e tipagem
make lint
make typecheck
```
