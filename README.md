
# RIMA: Riemannian Intrinsic Multimodal Alignment for Clinico-Genomic Survival Prediction
<img width="15003" height="7773" alt="fig2" src="https://github.com/user-attachments/assets/2ab4ac93-486b-41a6-987a-7500d6f6eb96" />

This repository provides the official implementation of **RIMA**, a geometry-aware multimodal survival prediction framework for integrating whole slide images (WSIs) and genomic profiles.

RIMA formulates clinico-genomic survival prediction as a **structure-preserving multimodal alignment** problem. Instead of directly concatenating visual and genomic features in a Euclidean latent space, RIMA learns modality-specific Riemannian metric tensors and performs topology-aware Gromov-Wasserstein alignment between WSI patch features and pathway-level genomic anchors.

---

## Overview

Whole slide images and genomic profiles provide complementary evidence for cancer prognosis. However, they have highly heterogeneous structures:

- WSIs contain thousands of spatially distributed tissue patches.
- Gene expression profiles contain high-dimensional molecular signals.
- Direct fusion may cause modality dominance and structural mismatch.

RIMA addresses these challenges through the following components:

1. **Pathway-Aware Genomic Encoding (PAGE)**  
   Raw gene expression profiles are transformed into pathway-level genomic anchors using a binary pathway-gene incidence matrix.

2. **Riemannian Metric Learning**  
   Visual patch features and genomic anchors are projected into a shared geometric space and equipped with learnable SPD metric tensors.

3. **Entropic Gromov-Wasserstein Alignment**  
   A topology-aware transport plan is learned between WSI patches and pathway anchors based on intrinsic Riemannian cost matrices.

4. **Geometry-Preserving Regularization**  
   Log-Euclidean metric regularization and complex-valued conformal consistency regularization are used to reduce geometric distortion.

5. **Survival Prediction**  
   Transported visual features and genomic anchors are integrated by gated fusion for Cox-based survival risk prediction.

---

## Repository Structure

```text
RIMA-main/
├── main.py                         # Training and evaluation entry
├── model.py                        # RIMA model implementation
├── ExactSizeSurvivalDataset.py     # Dataset loader for WSI features and genomics
├── README.md
└── wsi_core/
    ├── create_patches_fp.py        # WSI tissue segmentation and patch extraction
    ├── extract_features_fp.py      # Patch-level feature extraction
    ├── dataset_modules/
    │   ├── dataset_generic.py
    │   ├── dataset_h5.py
    │   └── wsi_dataset.py
    └── wsi_core/
        ├── WholeSlideImage.py
        ├── batch_process_utils.py
        ├── file_utils.py
        ├── util_classes.py
        └── wsi_utils.py
````

---

## Requirements

The code is implemented in Python and PyTorch.

Recommended environment:

```bash
conda create -n rima python=3.9 -y
conda activate rima
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy pandas h5py tqdm lifelines tensorboard matplotlib scikit-learn pillow timm openslide-python
```

For WSI processing, OpenSlide is required.

Ubuntu:

```bash
sudo apt-get install openslide-tools
```

Conda:

```bash
conda install -c conda-forge openslide
```

---

## Data Preparation

RIMA requires three types of inputs:

1. WSI patch-level features saved as `.pt` files.
2. Genomic expression profiles saved as `.csv`.
3. Survival labels and fold split files.

A recommended data organization is:

```text
DATA/
└── COAD/
    ├── feat/
    │   ├── TCGA-XX-XXXX-01Z-00-DX1.pt
    │   └── ...
    └── h5_files/
        ├── TCGA-XX-XXXX-01Z-00-DX1.h5
        └── ...

csv/
├── COAD.csv
└── COAD/
    ├── fold_1.csv
    ├── fold_2.csv
    ├── fold_3.csv
    ├── fold_4.csv
    └── fold_5.csv

gene/
├── COAD.csv
└── pathways_genes_matrix_new.csv
```

---

## Required CSV Formats

### 1. Survival label file

The survival label file, for example `csv/COAD.csv`, should contain:

```text
slide_id,survival_days,censorship
TCGA-XX-XXXX-01Z-00-DX1,1234,1
TCGA-YY-YYYY-01Z-00-DX1,856,0
```

where:

* `slide_id`: WSI slide ID.
* `survival_days`: observed survival time.
* `censorship`: event indicator used by the Cox loss. In this implementation, `1` indicates observed event and `0` indicates censored case.

### 2. Fold split files

Each fold file, such as `csv/COAD/fold_1.csv`, should contain:

```text
train,val,test
TCGA-XX-XXXX-01Z-00-DX1,TCGA-AA-AAAA-01Z-00-DX1,TCGA-BB-BBBB-01Z-00-DX1
...
```

### 3. Genomic expression file

The genomic expression file, for example `gene/COAD.csv`, should contain gene expression values indexed by gene names. In the dataset loader, the file is read by:

```python
pd.read_csv(genomics_txt, index_col=0).T
```

Therefore, after transposition, the index should correspond to patient IDs, such as the first 15 characters of TCGA slide IDs.

### 4. Pathway-gene incidence matrix

The pathway-gene matrix, for example `gene/pathways_genes_matrix_new.csv`, should be a binary matrix:

```text
Pathway,GeneA,GeneB,GeneC,...
Pathway_1,1,0,1,...
Pathway_2,0,1,0,...
```

Rows correspond to pathways and columns correspond to genes.
A value of `1` means that the gene belongs to the corresponding pathway.

---

## WSI Patch Extraction

The repository provides WSI preprocessing scripts adapted from CLAM-style workflows.

Example command:

```bash
python wsi_core/create_patches_fp.py \
  --source /path/to/raw_svs_files \
  --save_dir /path/to/save_patches \
  --patch_size 224 \
  --step_size 224 \
  --patch_level 0 \
  --patch \
  --seg \
  --stitch
```

This step generates tissue patches and coordinate files in `.h5` format.

---

## Patch Feature Extraction

After patch extraction, extract patch-level features using a pretrained pathology encoder.

Example command:

```bash
python wsi_core/extract_features_fp.py \
  --data_h5_dir /path/to/patch_h5_dir \
  --data_slide_dir /path/to/raw_svs_files \
  --csv_path /path/to/process_list_autogen.csv \
  --feat_dir /path/to/output_feature_dir \
  --model_name uni_v1 \
  --batch_size 256
```

Supported encoders in the script include:

```text
resnet50_trunc
uni_v1
conch_v1
```

The extracted WSI features are saved as `.pt` files and used as input to RIMA.

---

## Training

Example command for training and evaluating RIMA on the COAD cohort:

```bash
python main.py \
  --experiment_name RIMA \
  --MIL_model RIMA \
  --dataset COAD \
  --device_ids 0 \
  --fold_list 1 2 3 4 5 \
  --epochs 200 \
  --patience 20 \
  --lr_patience 8 \
  --max_lr 1e-4 \
  --min_lr 1e-6 \
  --csv_dir ./csv/COAD \
  --label_xlsx ./csv/COAD.csv \
  --feat_dir ./DATA/COAD/feat \
  --coords_dir ./DATA/COAD/h5_files \
  --genomics_txt ./gene/COAD.csv \
  --ckpt_dir ./ckpt/COAD \
  --logger_dir ./logger/COAD \
  --results_dir ./results/COAD
```

The model is trained with five-fold cross-validation.
For each fold, the model checkpoint is saved to:

```text
ckpt/COAD/RIMA/
```

TensorBoard logs are saved to:

```text
logger/COAD/RIMA/
```

Prediction results and Kaplan-Meier curves are saved to:

```text
results/COAD/RIMA/
```

---

## Model Details

The core model is implemented in `model.py`.

### Pathway-Aware Genomic Encoding

The `GenomicPathwayEncoder` converts raw gene expression into pathway-level anchors:

```python
masked_weight = self.weight * self.pathway_mask
pathway_scores = F.linear(x, masked_weight, self.bias)
anchors = feature_ext(pathway_scores)
```

This ensures that each pathway anchor is activated only by genes belonging to that pathway.

### Riemannian Metric Learning

The `RiemannianMetricNet` predicts a lower triangular matrix and constructs an SPD metric tensor:

```python
G = L @ L.T + eta * I
```

The learned metric tensor is used to compute intrinsic Riemannian distances between visual patches or genomic anchors.

### Entropic Gromov-Wasserstein Alignment

The `RiemannianIntrinsicGWAlignment` module computes intrinsic cost matrices and solves the transport plan using Sinkhorn iterations.

```python
T = gw_aligner(v_geo, g_geo, G_V, G_G)
```

The learned transport plan maps visual patch features into the genomic anchor space.

### Geometry Regularization

The geometric regularization contains:

* Log-Euclidean metric regularization between visual and genomic metric tensors.
* Complex-valued conformal consistency loss between transported visual geometric embeddings and genomic embeddings.

```python
loss_metric = ||log(G_V) - log(G_G)||_F
loss_holo = mean(|T^T z_V - z_G|^2)
```

### Survival Prediction

Transported visual features and genomic anchors are fused using a gated fusion module:

```python
fused = gate * h_v + (1 - gate) * h_g
z_final = concat(fused, h_v, h_g)
```

A linear prediction head outputs the patient-level risk score.

---

## Evaluation

The main evaluation metric is the Concordance Index (C-index):

```python
concordance_index(time, -exp(predicted_risk), event)
```

The code also exports patient-level risk scores and plots Kaplan-Meier curves for high-risk and low-risk groups.

Output files include:

```text
survival_1.csv
survival_2.csv
...
km_1.jpg
km_2.jpg
...
```

Each `survival_x.csv` contains:

```text
slide_id,survival_days,event_status,predicted_risk
```

---

## Important Notes

1. **Pathway matrix path**

   In `model.py`, the default pathway matrix path is currently set as:

   ```python
   csv_path=r'S:\code\TPAMI\gene\pathways_genes_matrix_new.csv'
   ```

   Please modify it to your local path, for example:

   ```python
   csv_path='./gene/pathways_genes_matrix_new.csv'
   ```

2. **Feature dimension**

   The current implementation uses a two-layer MLP to project 1024-dimensional WSI features into a 512-dimensional visual feature space:

   ```python
   nn.Linear(1024, 1024)
   nn.ReLU()
   nn.Linear(1024, 512)
   ```

   If your extracted features have a different dimension, please modify the MLP input dimension accordingly.

3. **Utility imports**

   The training script uses utility functions for data splitting, early stopping, and Kaplan-Meier plotting. If you use this repository as a standalone project, please make sure these utility modules are available or update the import paths accordingly.

4. **Patch feature directory**

   The dataset loader searches for files using:

   ```python
   glob(os.path.join(feat_dir, slide_id + '.pt*'))
   ```

   Please make sure that `--feat_dir` points to the folder containing the `.pt` feature files.

---



---

## Acknowledgements

This repository uses WSI preprocessing and feature extraction utilities inspired by open-source computational pathology pipelines such as CLAM. We thank the authors of these projects for making their code publicly available.

---

## License

This project is released for academic research purposes only. Please contact the authors for commercial use.

```

