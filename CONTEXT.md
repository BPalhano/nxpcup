# 🏎️ NXP Cup Autonomous Driving — Technical Context & Knowledge Base (`CONTEXT.md`)

> **Nota para LLMs e Engenheiros:** Este documento consolida todo o conhecimento de engenharia, arquitetura de hardware e software, formulações matemáticas de controle cinemático, topologia de pista e especificações de modelos neurais do projeto **NXP Cup Autonomous Driving** no ecossistema `nxpcup`. Use este contexto para continuar o desenvolvimento, treinar modelos ou debugar pipelines sem perda de coerência técnica.

---

## 1. 📌 Visão Geral do Projeto & Arquitetura de Ecossistema

O projeto visa dotar um veículo autônomo em escala 1:10 (kart de corrida) de capacidade de navegação em tempo real em uma pista padronizada da competição internacional **NXP Cup** (antiga *Freescale Cup*).

O ecossistema divide-se em dois projetos complementares e sinérgicos:
1. **`nxpcup_database` (`~/Documentos/Personal/nxpcup_database`):**
   - Repositório do dataset mestre preparado para publicação no Kaggle.
   - Contém as 3.433 imagens padronizadas em $200 \times 66 \times 3$.
   - Contém os segmentadores de alta resolução (`PrecisionTrackSegmenter`, `NXPUNetDeep`, `NXPU2NetDeep`, `NXPU2NetLite`).
   - Gera o arquivo mestre de anotações [`segmented_labels.csv`](file:///home/eulus/Documentos/Personal/nxpcup/main/data/processed/segmented_labels.csv).
2. **`nxpcup` (`~/Documentos/Personal/nxpcup`):**
   - Repositório principal de Engenharia e MLOps Edge AI.
   - Contém a esteira de dados com Albumentations (`main/data/dataset.py`).
   - Contém os modelos de regressão de esterçamento fim-a-fim (**MicroViT**, **NVIDIA DAVE-2**, **PalhanoCNN**).
   - Otimização com exportação ONNX e compilação TensorRT INT8 (< 3.5 ms de latência).
   - Monitoramento de Data Drift com Evidently AI, microsserviço FastAPI e empacotamento Docker Multi-Stage.

---

## 2. 🏎️ Especificações Físicas do Kart e Sensores

* **Plataforma Física:** Kart de corrida escala 1:10, tração traseira e direção dianteira esterçante baseada na geometria de Ackermann.
* **Intervalo de Esterçamento Mecânico:**
  $$\delta \in [-30.0^\circ, \, +30.0^\circ]$$
  *(Convenção de sinais: $\delta < 0$ para esterçamento à esquerda; $\delta > 0$ para esterçamento à direita; $\delta = 0.0^\circ$ estritamente em frente).*
* **Sensor Principal:** Câmera RGB montada centralizada sobre o para-choque frontal do kart, orientada para o solo.
* **Resolução Padronizada:** $200 \times 66 \times 3$ (formato canônico NVIDIA DAVE-2).
* **Taxa de Amostragem:** $10.0\text{ FPS}$ determinístico em 17 voltas gravadas na pista da **ENSICAEN** (Normandia, França).
* **Meta de Desempenho em Borda:** Latência de inferência $\le 3.5\text{ ms}$ ($> 250\text{ FPS}$) com consumo térmico reduzido em targets embarcados (NVIDIA Jetson Nano/Orin, NXP i.MX8M Plus, Raspberry Pi 5).

---

## 3. 🏁 Topologia da Pista e Normas da Competição NXP Cup

A pista oficial da NXP Cup possui geometria padronizada:

```text
       Borda Esquerda               Corredor Central               Borda Direita
       (Fita Preta 2.5 cm)         (Lona Branca Fosca)          (Fita Preta 2.5 cm)
              │                             │                            │
              ▼                             ▼                            ▼
      █████████████─────────────────────────────────────────────█████████████
      │◄── 2.5 cm ─►│◄────────────── 40 a 45 cm ───────────────►│◄── 2.5 cm ─►│
```

### Regras de Condução e Manobras
1. **Centralização no Corredor (Corridor Centering):** O veículo deve navegar baseado no **centro do corredor** ($x_C = (x_L + x_R) / 2$), nunca se prendendo ou seguindo uma única fita isolada.
2. **Retas Simples (`STRAIGHT`):** Filtradas com banda morta (*deadband*) de $|\delta| < 3.2^\circ$ e $|e_{\text{lat}}| < 7.0\text{ px} \implies 0.00^\circ$ para evitar oscilações de esterçamento (*bang-bang limit cycles*).
3. **Curvas Acentuadas e Entradas Quase Perpendiculares (`LEFT_CURVE` / `RIGHT_CURVE`):** Quando o carro entra em curvas fechadas ou chicanes com ângulo quase perpendicular às guias pretas, o controle cinemático calcula o vetor de mira (*lookahead waypoint*) somado à correção de erro lateral, saturando no limite físico ($\pm 30.0^\circ$) para forçar a recuperação rápida do centro da pista.
4. **Cruzamentos em 90° (`CROSSROAD_OR_PEDESTRIAN`):** Interseções perpendiculares geram faixas pretas horizontais. A regra da NXP Cup exige manter trajetória rigorosamente reta ($0.00^\circ$).
5. **Faixas de Largada/Pedestre (`CROSSROAD_OR_PEDESTRIAN`):** Marcadores transversais listrados múltiplos são detectados e navegados a $0.00^\circ$.

---

## 4. 📐 Formulação Matemática do Ângulo de Esterçamento do Próximo Passo

### 4.1. Transformação de Perspectiva Inversa (IPM / Bird's Eye View)
$$\begin{bmatrix} x_{\text{bev}} \\ y_{\text{bev}} \\ 1 \end{bmatrix} \sim \mathbf{H}_{3 \times 3} \begin{bmatrix} x_{\text{cam}} \\ y_{\text{cam}} \\ 1 \end{bmatrix}$$

Pontos calibrados no sensor do kart:
* Câmera: $[(32, 30), (168, 30), (198, 63), (2, 63)]$
* BEV ($400 \times 400$): $[(60, 0), (340, 0), (340, 400), (60, 400)]$

### 4.2. Extração da Linha Central ($x_C(y)$)
Para cada fatia no espaço métrico BEV, extraem-se os limites $x_L(y)$ e $x_R(y)$ do corredor branco. O centro matemático da pista é:
$$x_C(y) = \frac{x_L(y) + x_R(y)}{2}$$
Ajusta-se um polinômio quadrático $x_C(y) = a y^2 + b y + c$.

### 4.3. Lei de Controle Cinemático para Centralização no Próximo Passo (Stanley + Pure Pursuit)
O label principal do dataset (`next_step_steer_deg` / `angle_deg`) é formulado como:

1. **Posição Atual do Veículo:**
   $$P_{\text{car}} = (x_{\text{car}}, y_{\text{car}}) = (200, 360)$$
2. **Erro Lateral Imediato no Para-Choque (Cross-Track Error):**
   $$e_{\text{lat}} = x_C(y_{\text{base}}) - x_{\text{car}}$$
   *(Distância métrica transversal em pixels do centro do carro até o centro da pista).*
3. **Ponto de Mira do Próximo Passo (Lookahead Target Waypoint):**
   Calcula-se o ponto central da pista a uma distância de antecipação $\Delta y_{\text{look}}$:
   $$x_{\text{target}} = x_C(y_{\text{look}}), \quad y_{\text{target}} = y_{\text{look}}$$
4. **Vetor de Deslocamento Corretivo Combinado:**
   $$\Delta x_{\text{comb}} = (x_{\text{target}} - x_{\text{car}}) + k_{\text{lat}} \cdot e_{\text{lat}}$$
   $$\Delta y = y_{\text{base}} - y_{\text{look}}$$
5. **Ângulo de Esterçamento do Próximo Passo ($\delta_{\text{next\_step}}$):**
   $$\theta_{\text{raw}} = \arctan2(\Delta x_{\text{comb}}, \, \Delta y)$$
   $$\delta_{\text{next\_step}} = \text{clip}\left(\theta_{\text{raw}} \cdot K_{\text{gain}}, \, -30.0^\circ, \, +30.0^\circ\right)$$
   * Se o carro estiver no centro de uma reta: $\Delta x_{\text{comb}} = 0 \implies \delta = 0.00^\circ$.
   * Se o carro estiver deslocado para a direita: $e_{\text{lat}} < 0 \implies \delta < 0$ (esterça à esquerda para centralizar).
   * Se a pista for **quase perpendicular** (curva fechada): $\Delta x$ é muito maior que $\Delta y$, e a função $\arctan2$ satura em $\pm 30.0^\circ$, gerando esterçamento máximo para trazer o kart de volta ao eixo da pista.

---

## 5. 🧠 Modelos Neurais do Projeto

### 5.1. Modelos de Regressão Fim-a-Fim (`main/src/models/`)
1. **`MicroViT` (Proposto - `main/src/models/micro_vit.py`):**
   * Stem Convolucional ($3 \times 3$, stride 2) reduzindo a $16 \times 50 \times 64$.
   * Patch Embedding ($2 \times 2$, 200 tokens) + 2 Blocos Transformer com Multi-Head Self-Attention ($d_{\text{model}} = 128$, 4 heads, MLP=256).
   * Global Average Pooling + Camada Linear com projeção contínua $\to \hat{y} \in [-1.0, 1.0]$ ($\pm 30.0^\circ$).
   * **Total de parâmetros reais:** **359.841 parâmetros** (dentro do budget de 350k-450k; $< 41\%$ do paper e $< 23\%$ do DAVE-2).
2. **`NVIDIA DAVE-2` (Baseline - `main/src/models/dave2.py`):**
   * 5 camadas convolucionais ($3 \times 5 \times 5$ stride 2 + $2 \times 3 \times 3$ stride 1) + 5 camadas lineares ($1152 \to 1164 \to 100 \to 50 \to 10 \to 1$).
   * **Total de parâmetros reais:** **1.595.511 parâmetros** (Bojarski et al., 2016).
3. **`PalhanoCNN` (Legado - `main/src/models/palhano_cnn.py`):**
   * CNN de 5 convoluções + 4 camadas densas ($1152 \to 512 \to 256 \to 128 \to 1$) do paper original da ENSICAEN.
   * **Total de parâmetros reais:** **exatamente 886.037 parâmetros** (paridade matemática com o paper).
4. **`ModelFactory` (`main/src/models/factory.py`):**
   * Factory Pattern para carregamento dinâmico (`dave2`, `palhano_cnn`, `micro_vit`).

### 5.2. Modelos de Segmentação de Pista (`src/segmentation/unet_models.py`)
1. **`NXPUNetDeep` (16.2M parâmetros):**
   * 5 estágios encoder-decoder ($32 \to 64 \to 128 \to 256 \to 512$).
   * Bottleneck com ASPP (dilatações $r \in [1, 2, 4, 8]$).
   * Portas de Atenção Squeeze-and-Excitation (SE-Attention) em todas as skip connections.
2. **`NXPU2NetDeep` (10.7M parâmetros):**
   * 6 estágios de blocos aninhados `RSU-7`, `RSU-6`, `RSU-5`, `RSU-4`, `RSU-4F`, `RSU-4F`.
   * Supervisão profunda sêxtupla fundida por convolução $1 \times 1$.
3. **`NXPUNet` (1.9M parâmetros):** U-Net compacta de 4 níveis para Jetson Nano.
4. **`NXPU2NetLite` (1.0M parâmetros):** U²-Net ultraleve para NPUs embarcadas (< 3.5 ms).

---

## 6. 📊 Dicionário de Dados (`labels_continuous.csv` e `segmented_labels.csv`)

O arquivo [`labels_continuous.csv`](file:///home/eulus/Documentos/Personal/nxpcup/main/data/processed/labels_continuous.csv) e [`segmented_labels.csv`](file:///home/eulus/Documentos/Personal/nxpcup/main/data/processed/segmented_labels.csv) contém as 3.433 imagens:

| Coluna | Tipo | Descrição |
| :--- | :--- | :--- |
| `filename` | string | Nome do arquivo em `main/data/processed/images/` |
| `video_id` | string | Identificador da volta/vídeo (17 voltas gravadas) |
| `frame_idx` | int | Índice sequencial do frame no vídeo |
| `timestamp_s` | float | Timestamp no vídeo em segundos (10.0 FPS determinístico) |
| `angle_deg` | float | **Ângulo de esterçamento para centralização do próximo passo $[-30^\circ, +30^\circ]$** |
| `angle_norm` | float | Ângulo normalizado em $[-1.0, 1.0]$ (`angle_deg / 30.0`) |
| `next_step_steer_deg` | float | **Label de controle exato (Stanley + Pure Pursuit)** |
| `next_step_steer_norm`| float | Ângulo normalizado de próximo passo $[-1.0, 1.0]$ |
| `gt_angle_deg` | float | Ângulo original da baseline |
| `curvature` | float | Curvatura $\kappa$ da trajetória de centralização |
| `lateral_offset_px` | float | Erro lateral no para-choque em pixels (cross-track error) |
| `track_width_px` | float | Largura medida do corredor métrico BEV em pixels |
| `near_angle_deg` | float | Tangente de curvatura na entrada da curva |
| `mid_angle_deg` | float | Tangente de curvatura no ápice |
| `far_angle_deg` | float | Tangente de curvatura na saída da curva |
| `mean_curve_angle_deg` | float | Média integrada da curvatura ao longo do trecho |
| `confidence` | float | Score de confiança geométrica $[0.15, 1.00]$ |
| `mode` | string | Elemento da pista: `STRAIGHT`, `LEFT_CURVE`, `RIGHT_CURVE`, `FALLBACK_UNCERTAIN` |
| `class_name` | string | Classe normalizada em minúsculas para estratificação de dados |
| `section_id` | string | ID único do trecho físico contínuo (ex.: `video_01_sec_002_LEFT_CURVE`) |
| `section_mean_steer_deg` | float | **Média do esterçamento em todo o trecho físico contínuo** |
| `section_std_steer_deg` | float | Desvio padrão do esterçamento no trecho |
| `section_frame_count` | int | Total de frames que compõem o trecho físico |

---

## 7. 🚀 Integração no PyTorch (`main/data/dataset.py`)

```python
from main.data.dataset import NXPCupDataset, load_dataset_records, get_train_transforms

records = load_dataset_records("main/data/processed/labels_continuous.csv")
dataset = NXPCupDataset(
    records=records,
    images_dir="main/data/processed/images",
    transforms=get_train_transforms(),
    is_train=True,
)

sample = dataset[0]
print("Shape da Imagem :", sample["image"].shape)               # torch.Size([3, 66, 200])
print("Ângulo Próx Step:", sample["next_step_steer_deg"].item()) # ex: -14.35°
print("Erro Lateral    :", sample["lateral_offset"].item())       # ex: -5.2 px
```

---

## 8. 🏎️ Pipeline Unificado de Treinamento e MLOps (`main/src/train.py`)

A esteira de treinamento unifica os três modelos neurais através do Factory Pattern, Huber Loss e MLflow Tracking:

```bash
# Treinamento via Makefile
make train-microvit     # Executa MicroViT com parâmetros ótimos
make train-dave2        # Executa Baseline NVIDIA DAVE-2
make train-palhano      # Executa Baseline PalhanoCNN ENSICAEN

# Execução customizada via CLI
uv run --directory main python -m src.train --model micro_vit --epochs 35 --batch-size 64
```

### Artefatos de Avaliação Gerados
- `best_model.pt`: Checkpoint do modelo de menor MAE em validação.
- `learning_curves.png`: Curvas de Huber Loss e MAE (°).
- `steer_scatter.png`: Dispersão de predições vs. Ground Truth com reta identidade ($y = x$).
- `residual_distribution.png`: Distribuição gaussiana dos resíduos com média e limites $\pm 1\sigma$.
- `evaluation_summary.json`: Métricas consolidadas (MAE, RMSE, Max Error, $R^2$).

---

## 9. 🏁 Status Atual do Projeto e Roadmap de Sprints

| Sprint | Escopo | Status |
| :--- | :--- | :---: |
| **Sprint 0 & 1** | Setup, amostragem determinística 10 FPS, segmentação BEV, labels cinemáticos | ✅ Concluído |
| **Sprint 2** | DAVE-2, PalhanoCNN (886k), MicroViT (359k), ModelFactory, MLflow Pipeline | ✅ Concluído |
| **Sprint 3** | Exportação ONNX, simplificação onnxsim, calibração INT8 TensorRT, benchmarks | ✅ Concluído (P50 2.16 ms, 436.7 FPS) |
| **Sprint 4** | Monitoramento de Data Drift com Evidently AI (voltas normais vs. com sombras) | ✅ Concluído (MicroViT degrada apenas +19.7% vs +80.2% DAVE-2) |
| **Sprint 5** | Microsserviço FastAPI (`POST /steer`), empacotamento Docker multi-stage, README | ⏳ Próximo |

