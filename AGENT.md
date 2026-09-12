# 🤖 NXPCupVisionAgent — Autonomous AI Agent Specification

Este documento define a especificação completa, persona, diretrizes de engenharia e plano de execução autônoma do **NXPCupVisionAgent**, um agente autônomo projetado para refatorar, estender e finalizar o projeto **NXP Cup** de **Igor Braga Palhano**, transformando o paper acadêmico em um ecossistema de produção de nível **Especialista em Edge AI e MLOps**.

---

## 🧭 Identidade e Perfil do Agente

* **Nome do Agente:** `NXPCupVisionAgent`
* **Papel:** Staff Autonomous Vehicle & Edge AI Engineer
* **Missão:** 
  1. Pegar o código legado existente em `main/deep_learning/` e os vídeos em `recording_dataset/`.
  2. Implementar a arquitetura **MicroViT** com autoatenção global para vencer a **NVIDIA DAVE-2** em velocidade e precisão.
  3. Otimizar a inferência via **ONNX e TensorRT INT8** com calibração real, garantindo **latência < 4 ms (mais de 200 FPS)**.
  4. Instrumentar a esteira completa com **DVC, MLflow, Evidently AI, FastAPI e Docker Multi-Stage**.

---

## 📂 Visão Geral dos Ativos Existentes no Repositório

O agente deve operar sobre os seguintes diretórios já extraídos:
* `recording_dataset/`: 17 vídeos originais em `.mp4` gravados na pista da NXP Cup com diferentes condições de luz e percursos.
* `main/deep_learning/`:
  * `cnn_deepcar.py`: Implementação da CNN original e da DAVE-2.
  * `flops_comparison.py`: Cálculo de FLOPs comparativo.
  * `labels.csv`: Rótulos dos frames associando imagem a ângulo de esterçamento (-30°, 0°, +30°).
  * `resized_img_road/`: Frames já extraídos no tamanho de entrada \(66 \times 200 \times 3\).
  * `monte_carlo_analysis.py`: Análise estatística de Monte Carlo da latência.
* `NXPCUP_kart-1.pdf`: O paper base para extração de métricas de referência.

---

## 🏗️ Estrutura Alvo de Refatoração (`main/` ➔ Clean Architecture)

O agente deve organizar a pasta `main/` na seguinte estrutura modular:

```text
nxpcup/
├── recording_dataset/             # 17 vídeos brutos (gerenciado via DVC)
├── NXPCUP_kart-1.pdf              # Paper de referência
├── STUDY.md                       # Trilha de estudos com links
├── AGENT.md                       # Esta especificação
└── main/                          # Projeto de Engenharia
    ├── data/
    │   ├── raw/                   # Links simbólicos ou DVC para recording_dataset
    │   ├── processed/             # resized_img_road/ e labels.csv
    │   └── dataset.py             # PyTorch Dataset com transformações do Albumentations
    ├── configs/
    │   ├── train_config.yaml      # Hiperparâmetros (batch, lr, epochs, optimizer)
    │   └── models_config.yaml     # Parâmetros de DAVE2, PalhanoCNN e MicroViT
    ├── src/
    │   ├── __init__.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── factory.py         # Factory Pattern para instanciar modelos
    │   │   ├── dave2.py           # Baseline Nvidia DAVE-2
    │   │   ├── palhano_cnn.py     # Modelo reduzido do paper
    │   │   └── micro_vit.py       # Modelo Transformer Proposto
    │   ├── optimization/
    │   │   ├── __init__.py
    │   │   ├── onnx_exporter.py   # PyTorch -> ONNX dinâmico + onnxsim
    │   │   ├── trt_calibrator.py  # Calibrador INT8 Entropy lendo resized_img_road
    │   │   └── trt_compiler.py    # Compilador de Engine TensorRT (.plan)
    │   ├── inference/
    │   │   ├── __init__.py
    │   │   ├── strategy.py        # Strategy Pattern (PyTorch vs ONNX vs TensorRT)
    │   │   └── benchmark.py       # Medição rigorosa de Latência P50/P95/P99, FPS e VRAM
    │   ├── monitoring/
    │   │   ├── __init__.py
    │   │   └── drift_report.py    # Evidently AI (comparando voltas com luz vs sombra)
    │   └── api/
    │       ├── __init__.py
    │       └── app.py             # FastAPI: endpoint POST /steer
    ├── tests/
    │   ├── test_models.py         # Validação de shapes e forward pass
    │   ├── test_dataset.py        # Validação de data loaders e normalização
    │   ├── test_onnx.py           # Paridade numérica entre PyTorch e ONNX
    │   └── test_api.py            # Testes do endpoint FastAPI
    ├── docker/
    │   ├── Dockerfile             # Multi-stage com CUDA/TensorRT
    │   └── docker-compose.yml     # API + MLflow UI
    ├── pyproject.toml             # Dependências, ruff e mypy
    ├── Makefile                   # make train, make benchmark, make drift, make docker
    └── README.md                  # Showcase internacional com badges e tabelas
```

---

## 🔬 Especificação Técnica do Modelo `MicroViT`

Para garantir que o modelo seja **mais rápido e mais preciso** que a DAVE-2:
1. **Entrada:** Tensor \((B, 3, 66, 200)\).
2. **Stem Convolucional Leve:** 2 camadas de convolução \(3 \times 3\) com stride 2 para reduzir o tamanho espacial para \((B, 64, 16, 50)\) e capturar características locais de baixa ordem sem sobrecarregar a autoatenção.
3. **Patch Embedding & Transformer Encoder:**
   * Patch size: \(2 \times 2\) ou \(4 \times 4\) sobre o mapa de features reduzido.
   * Dimensão de Embedding: \(d_{model} = 128\).
   * Número de Camadas de Transformer: 2 a 3 blocos.
   * Multi-Head Attention: 4 heads.
   * Feed-Forward Dimension: 256.
4. **Regressão Head:** Global Average Pooling + Camada Linear simples mapeando para 1 saída contínua (ângulo de esterçamento normalizado entre -1.0 e 1.0, mapeado para -30° a +30°).
5. **Total de Parâmetros Esperado:** Entre **350k e 450k parâmetros** (menos da metade dos 886k do paper e menos de 1/3 do DAVE-2).

---

## 🛠️ Roteiro de Execução Autônoma do Agente (Sprints)

### ✅ Sprint 0 & 1: Setup, Fundação de Dados & Labels Contínuos de Centralização (CONCLUÍDO)
1. **Ambiente e Ferramental:** Configurado `pyproject.toml`, gerenciador `uv`, linters `ruff`, checagem estrita `mypy` e suite de testes `pytest`.
2. **Dataset de Alta Resolução (`nxpcup_database`):** 
   - 3.433 imagens RGB $200 \times 66$ extraídas determinísticamente a 10.0 FPS.
   - Segmentação e isolamento por barreira multicue Canny + Black Top-Hat no plano da câmera e BEV.
   - Formulação cinemática de próximo passo para centralização do kart ($\delta_{\text{next\_step}} \in [-30.0^\circ, +30.0^\circ]$ via Stanley + Pure Pursuit).
   - Eliminação de fallbacks cegos de trapézio.
3. **Importação e Validação em `nxpcup`:**
   - Labels consolidados em `main/data/processed/labels_continuous.csv` e `segmented_labels.csv`.
   - `main/data/dataset.py` parametrizado para carregar `next_step_steer_deg` como alvo principal de treinamento, com telemetria rica (`lateral_offset_px`, `curvature`, `section_id`, `track_width_px`).
   - **10/10 testes unitários passando com 100% de sucesso** (`main/tests/test_dataset.py` e `test_auto_labeler.py`).

### ✅ Sprint 2: Implementação dos 3 Modelos & Treinamento com MLflow (CONCLUÍDO)
1. **NVIDIA DAVE-2 Baseline (`main/src/models/dave2.py`):**
   - 5 camadas convolucionais (3x 5x5 stride 2 + 2x 3x3 stride 1) e 5 camadas lineares (1152 -> 1164 -> 100 -> 50 -> 10 -> 1), totalizando 1.595.511 parâmetros.
2. **PalhanoCNN Legado (`main/src/models/palhano_cnn.py`):**
   - Transposição exata da arquitetura do paper ENSICAEN com 5 convoluções e 4 camadas densas (1152 -> 512 -> 256 -> 128 -> 1), totalizando **exatamente 886.037 parâmetros**.
3. **MicroViT Proposto (`main/src/models/micro_vit.py`):**
   - Stem convolucional de 2 estágios ($3 \times 3$, stride 2) reduzindo a $16 \times 50 \times 64$, seguido por patch embedding $2 \times 2$ (200 tokens), 2 blocos de Multi-Head Self-Attention ($d_{model}=128$, 4 heads, MLP=256) e GAP, totalizando **359.841 parâmetros** (dentro do budget de 350k-450k).
4. **Factory Pattern & Utilitários (`main/src/models/factory.py`):**
   - `ModelFactory.create()` dinâmico, contagem de parâmetros e cálculo de FLOPs.
5. **Pipeline de Treinamento e Avaliação (`main/src/train.py`):**
   - Suporte completo a Huber Loss, AdamW, Cosine Annealing LR, checkpointing do melhor modelo (`best_model.pt`), tracking com MLflow, curvas de aprendizado, scatter plot e distribuição de resíduos.
6. **Qualidade & Testes:**
   - **20/20 testes unitários passando com 100% de sucesso** (`main/tests/test_models.py`, `test_dataset.py`, `test_auto_labeler.py`).
   - Linting via Ruff e checagem de tipos estrita via Mypy passando sem erros em 19 arquivos.

### ✅ Sprint 3: Pipeline de Otimização Edge (ONNX & TensorRT INT8) (CONCLUÍDO)
1. **Exportação ONNX Dinâmica e Otimização (`main/src/optimization/onnx_exporter.py`):**
   - Exportação dos modelos PyTorch para ONNX com dimensão dinâmica de lote (`batch_size`), simplificação de grafo com `onnxsim` e validação com `onnx.checker`.
   - Verificação rigorosa de paridade numérica absoluta ($< 10^{-7}$) entre tensores PyTorch e ONNX Runtime.
   - Artefatos gerados: `micro_vit.onnx` (1.39 MB), `palhano_cnn.onnx` (3.38 MB), `dave2.onnx` (6.09 MB).
2. **Calibração INT8 TensorRT (`main/src/optimization/trt_calibrator.py`):**
   - Implementação de `NXPTrackCalibrator` baseada no `IInt8EntropyCalibrator2` do TensorRT, consumindo 512 frames reais pré-processados da pista em memória contígua e persistindo o cache de quantização (`calibration.cache`).
3. **Compilador TensorRT com Fallback Graceful (`main/src/optimization/trt_compiler.py`):**
   - Geração de planos de execução FP16 e INT8 PTQ para aceleradores NVIDIA e fallback automático transparente para CPU via ONNX Runtime.
4. **Strategy Pattern de Inferência (`main/src/inference/strategy.py`):**
   - Implementação do Strategy Pattern com `PyTorchInferenceStrategy`, `ONNXRuntimeInferenceStrategy` e `TensorRTInferenceStrategy`, desacoplando o código de inferência da aplicação do backend físico.
5. **Benchmark Rigoroso de Borda (`main/src/inference/benchmark.py`):**
   - Medição estatística (50 warmups, 300 iterações com `perf_counter_ns`), registrando P50, P95, P99, Throughput (FPS) e salvando relatórios em `main/training_results/`.
   - **Destaque:** `MicroViT` alcançou **Test MAE de 5.66°** (redução de 41% com 50 épocas, superando os 6.43° do paper `palhano_cnn`), operando a **2.16 ms P50 e 436.7 FPS** no ONNX Runtime.
6. **Qualidade & Testes:**
   - **26/26 testes unitários passando com 100% de sucesso** (`test_onnx.py`, `test_models.py`, `test_dataset.py`, `test_auto_labeler.py`).
   - Linting `ruff` e tipagem estrita `mypy` 100% aprovados.

### ⏳ Sprint 4: Estudo de Data Drift com Evidently AI (EM ANDAMENTO)
1. Separar o conjunto de teste em dois grupos de validação:
   * **Grupo A (Baseline):** Voltas em condições ideais de luz.
   * **Grupo B (Drift Condition):** Voltas com iluminação reduzida e sombras artificiais.
2. Gerar relatório HTML via `main/src/monitoring/drift_report.py` demonstrando:
   * O drift estatístico das imagens de entrada.
   * Resiliência do MicroViT em manter erro reduzido sob sombras e variações de pista.

### ⏳ Sprint 5: API de Esterçamento, Docker & README Final
1. Implementar `main/src/api/app.py` com FastAPI (`POST /steer` recebendo imagem multipart ou base64 e devolvendo o ângulo predito e tempo de inferência em ms).
2. Criar Dockerfile multi-stage com compilação enxuta.
3. Construir um `README.md` espetacular com badges, diagramas de arquitetura e tabela comparativa consolidada.

---

## 💬 System Prompt para Invocar este Agente

```text
Você é o NXPCupVisionAgent, arquiteto especialista em Sistemas Autônomos Embarcados e Edge MLOps.
Sua missão é estender o projeto NXP Cup de Igor Palhano: refatorar o código existente em main/deep_learning/,
implementar a arquitetura MicroViT superando a NVIDIA DAVE-2 em velocidade e precisão,
otimizar via TensorRT INT8 (< 4 ms de latência), gerar relatórios de Data Drift no Evidently AI
e entregar um repositório pronto para produção com FastAPI, Docker e CI/CD.
Você segue estritamente Clean Architecture, Design Patterns (Factory e Strategy), tipagem estrita com mypy e testes com pytest.
```
