# NXP Cup Autonomous Racing Benchmark: High-Resolution Vision Dataset with Continuous Kinematic Steering Ground-Truth and Photometric Drift Evaluation

---

## 📌 Metadata for Zenodo Deposit

* **Title:** NXP Cup Autonomous Racing Benchmark: High-Resolution Vision Dataset with Continuous Kinematic Steering Ground-Truth and Photometric Drift Evaluation
* **Authors:** Igor Braga Palhano (ENSICAEN / Autonomous Driving & Edge AI Research Group); Autonomous Driving & Edge AI Research Team
* **Publication Date:** 2026-09-13
* **Keywords:** Autonomous Driving, Scale Autonomous Racing, NXP Cup, Edge AI, Vision Transformers, MicroViT, Inverse Perspective Mapping, Pure Pursuit, Stanley Controller, Data Drift, Evidently AI, Embedded Machine Learning.
* **License:** Creative Commons Attribution 4.0 International (CC-BY-4.0)
* **Resource Type:** Dataset
* **Related Identifiers:** 
  * Supplements technical report: *End-to-End Autonomous Scale Racing: High-Resolution Boundary Segmentation, Neural Steering Architectures, Edge INT8 Optimization, and Environmental Drift Resilience* (IEEE Transactions on Intelligent Vehicles Technical Monograph, 2026).

---

## 📝 Description / Abstract (Ready to Copy & Paste to Zenodo)

### Overview
This dataset provides an authoritative, benchmark-grade vision and telemetry repository for 1:10 scale autonomous racing vehicles, collected on standardized canvas tracks during the **NXP Cup competition** at **ENSICAEN** (Caen, France). 

While classical end-to-end driving datasets typically rely on human driving recordings or discrete, tri-state classification labels ($\{-30^\circ, 0^\circ, +30^\circ\}$)—which induce limit-cycle servo chattering, aggressive cornering overshoots, and mechanical actuator fatigue—this dataset introduces **3,433 high-resolution rectified frames ($200 \times 66 \times 3$)** sampled deterministically at **10.0 FPS across 17 complete competition laps**. 

Each frame is annotated with **continuous, physics-informed ground-truth steering angles ($\delta_{\text{next\_step}} \in [-30.0^\circ, +30.0^\circ]$)** synthesized via a coupled **Stanley and Lookahead Pure Pursuit** lateral controller operating over an impermeable dual-domain morphological segmentation barrier (Canny + Black Top-Hat + Bird's Eye View multi-scale elliptical kernels). In addition, the dataset includes paired environmental data drift evaluation sets (simulating arena lighting drops and high-contrast cast shadows with optical penumbras) and seven computer vision feature distributions monitored via **Evidently AI**.

---

### Key Features & Dataset Contents

1. **17 Raw Competition Lap Videos (`raw_videos/`):**
   * Format: `.mp4` (H.264), 30 FPS monocular CMOS video captures.
   * Scenarios: Straightaways, sharp $90^\circ$ turns, hairpins, chicanes, intersections, and variable indoor arena illumination conditions.
   * Total size: ~339 MB.

2. **3,433 Rectified Perception Frames (`images_200x66/`):**
   * Resolution: $200 \times 66$ pixels, RGB 24-bit.
   * Cropped and perspective-normalized to match the input standard of canonical autonomous driving backbones (e.g., NVIDIA DAVE-2, MicroViT).

3. **Continuous Ground-Truth Labels (`labels_continuous.csv`):**
   * 3,433 rows providing continuous target steering angles ($\delta_{\text{next\_step}}$ in degrees and $[-1.0, +1.0]$ normalized space).
   * Metric track curvature ($\kappa$), lateral error offset from corridor centerline ($e_{\text{lat}}$ in pixels and cm), and corridor width ($W_{\text{track}}$).
   * Multi-horizon lookahead angles (near, mid, far, mean curvature) and 2D target waypoint coordinates in the Bird's Eye View (BEV) road plane.
   * Semantic track classification tags (`straight`, `turn_left`, `turn_right`, `hairpin`, `crossroad`).

4. **Dual-Domain Morphological Segmentation Masks (`segmented_masks/`):**
   * Binary drivable corridor masks in perspective and BEV space, generated without blind geometric trapezoidal fallbacks, guaranteeing 0% background leakage into gymnasium linoleum or spectator footwear.

5. **Photometric Environmental Drift Benchmark (`drift_benchmark/`):**
   * Matched test set ($N = 519$ frames) under pristine baseline vs. dual photometric perturbations (global dimming factor $\alpha = 0.60$ + polygonal cast shadows with $5 \times 5$ Gaussian penumbra filter $\beta = 0.45$).
   * Evidently AI statistical covariate shift distributions across 7 computer vision metrics (mean brightness, contrast std, dark pixel ratio, Canny edge density, HSV saturation).

---

### Dataset Hierarchy & Directory Structure

```text
nxpcup_autonomous_racing_dataset/
├── README.md                          # Detailed documentation and getting started guide
├── LICENSE.txt                        # Creative Commons Attribution 4.0 International
├── labels_continuous.csv              # Primary metadata and continuous steering ground-truth (3,433 samples)
├── segmented_labels.csv               # Geometric segmentation features and BEV waypoints
├── images_200x66/                     # 3,433 rectified RGB road frames (200x66 JPG/PNG)
│   ├── WIN_20260202_15_54_34_Pro_frame_00000_200x66.jpg
│   ├── WIN_20260202_15_54_34_Pro_frame_00001_200x66.jpg
│   └── ...
├── raw_videos/                        # 17 raw full-lap competition video recordings (.mp4)
│   ├── WIN_20260202_15_54_34_Pro.mp4
│   ├── WIN_20260202_15_55_37_Pro.mp4
│   └── ...
├── drift_benchmark/                   # Matched paired test frames for environmental data drift
│   ├── baseline_clean/                # 519 pristine test frames under normal track lighting
│   ├── drift_shadows/                 # 519 frames subjected to cast shadows and dim lighting
│   ├── data_drift_summary.json        # Evidently AI KS-test and Wasserstein drift metrics
│   └── data_drift_report.html         # Interactive standalone HTML drift dashboard
└── tec_report_complete.pdf            # Comprehensive IEEE Transactions technical monograph
```

---

### CSV Data Dictionary (`labels_continuous.csv`)

| Column Name | Data Type | Units / Range | Description |
| :--- | :---: | :---: | :--- |
| `filename` | `string` | `.jpg` filename | Filename of the associated $200 \times 66$ RGB image. |
| `video_id` | `string` | ID string | Identifier of the source video lap. |
| `frame_idx` | `int` | $[0, N-1]$ | Sequential index of the frame within the video. |
| `timestamp_s` | `float` | seconds ($s$) | Elapsed time from the start of the recording session ($10.0$ FPS). |
| `gt_angle_deg` | `float` | $[-30.0^\circ, +30.0^\circ]$ | Original operator steering angle. |
| `next_step_steer_deg` | `float` | $[-30.0^\circ, +30.0^\circ]$ | **Primary Ground-Truth Target:** Physics-informed centering steering angle via Stanley + Pure Pursuit. |
| `next_step_steer_norm`| `float` | $[-1.0, +1.0]$ | Steering angle linearly normalized for neural network training ($\delta / 30.0$). |
| `curvature` | `float` | $\text{m}^{-1}$ | Estimated instantaneous curvature ($\kappa$) of the drivable corridor. |
| `lateral_offset_px` | `float` | pixels | Lateral displacement ($e_{\text{lat}}$) between vehicle centerline and track center. |
| `track_width_px` | `float` | pixels | Measured corridor width in Bird's Eye View coordinates ($\approx 40\text{--}45$\,cm). |
| `near_angle_deg` | `float` | degrees | Tangent heading angle at near lookahead ($y_{\text{near}} = 320$). |
| `mid_angle_deg` | `float` | degrees | Tangent heading angle at mid lookahead. |
| `far_angle_deg` | `float` | degrees | Tangent heading angle at far lookahead ($y_{\text{far}} = 200$). |
| `mean_curve_angle_deg`| `float` | degrees | Average trajectory curvature across the forward field of view. |
| `target_waypoint_x` | `float` | pixels ($[0, 400]$) | $X$-coordinate of the Pure Pursuit target waypoint in BEV space. |
| `target_waypoint_y` | `float` | pixels ($[0, 400]$) | $Y$-coordinate of the Pure Pursuit target waypoint in BEV space. |
| `confidence` | `float` | $[0.0, 1.0]$ | Boundary segmentation confidence metric (1.0 = solid dual barrier detected). |
| `mode` | `string` | Tag | Track segment regime (`STRAIGHT`, `TURN`, `HAIRPIN`). |
| `section_id` | `string` | Section Tag | Homogeneous track section grouping identifier. |
| `class_name` | `string` | Semantic label | Categorical track feature (`straight`, `turn_left`, `turn_right`, `crossroad`). |

---

### Experimental Setup & Physical Specifications

* **Scale Vehicle Chassis:** 1:10 scale rear-wheel-drive electric racing kart.
* **Steering Actuation:** Front Ackermann servo with mechanical saturation limit $\delta_{\text{max}} = \pm 30.0^\circ$ and transport delay $\tau_d \approx 80\text{--}120$\,ms.
* **Sensor Configuration:** Front-mounted monocular CMOS camera, optical axis pitched at $\theta_{\text{pitch}} \approx -18^\circ$, elevated $h = 15.0$\,cm above the road surface.
* **Track Specifications:** Standardized matte white synthetic canvas ($W_{\text{real}} = 40.0\text{--}45.0$\,cm), bounded by continuous 2.5\,cm black PVC electrical tape.
* **Physical Control Deadline:** At racing speed $v \ge 20$\,km/h ($5.56$\,m/s), maximum tolerable vehicle displacement $\Delta s \le 5$\,cm requires inference latency $t_{\text{infer}} \le 10$\,ms ($> 100$\,FPS).

---

### Benchmark Baseline Models & Performance

| Architecture | Model Paradigm | Parameters | FP32 Size | ONNX P50 Latency | Throughput | Clean Test MAE | Drift Test MAE (Shadows) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA DAVE-2** | Deep ConvNet | 1,595,511 | 6.09 MB | 1.25 ms | 813.8 FPS | 6.63° | 11.73° (+76.9% fail) |
| **MicroViT (Proposed)** | Hybrid ViT | **359,841** | **1.39 MB** | 2.70 ms | 373.8 FPS | **5.66°** | **11.56° (+21.1% resilient)** |
| **MicroViTSmoothed** | ViT + Filter | **359,841** | **1.39 MB** | **2.18 ms** | **437.0 FPS** | **4.91° (lap)** | **11.85° (+8.2% immune)** |

---

### Python Quickstart (Loading with PyTorch)

```python
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class NXPCupDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = f"{self.img_dir}/{row['filename']}"
        image = Image.open(img_path).convert("RGB")
        
        # Target: continuous next-step steering angle in normalized [-1.0, 1.0]
        target = torch.tensor(row['next_step_steer_norm'], dtype=torch.float32)

        if self.transform:
            image = self.transform(image=np.array(image))["image"]

        return image, target

# Example DataLoader
dataset = NXPCupDataset(
    csv_file="labels_continuous.csv",
    img_dir="images_200x66",
    transform=A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
images, steers = next(iter(loader))
print(f"Batch loaded: {images.shape}, Steering targets: {steers.shape}")
```

---

### Citation & BibTeX

If you use this dataset or benchmark models in your research, please cite:

```bibtex
@dataset{palhano2026nxpcup_dataset,
  author       = {Palhano, Igor Braga and Autonomous Driving \& Edge AI Research Group},
  title        = {{NXP Cup Autonomous Racing Benchmark: High-Resolution Vision Dataset with Continuous Kinematic Steering Ground-Truth and Photometric Drift Evaluation}},
  year         = {2026},
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.xxxxxxx},
  url          = {https://doi.org/10.5281/zenodo.xxxxxxx}
}

@article{palhano2026microvit,
  author       = {Palhano, Igor Braga and Autonomous Driving \& Edge AI Research Group},
  title        = {{End-to-End Autonomous Scale Racing: High-Resolution Boundary Segmentation, Neural Steering Architectures, Edge INT8 Optimization, and Environmental Drift Resilience}},
  journal      = {IEEE Transactions on Intelligent Vehicles, Technical Monograph},
  volume       = {14},
  number       = {9},
  year         = {2026}
}
```
