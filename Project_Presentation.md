---
marp: true
theme: default
paginate: true
header: "PAIML Lab Project — Human Activity Recognition"
footer: "Kartikeya Jain & Ayush Yadav | Mentors: Dr. Pruthwik Mishra & Dr. Praveen Chandaliya"
style: |
  section {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 40px;
    font-size: 24px;
  }
  h1 {
    color: #1e3a8a;
  }
  h2 {
    color: #2563eb;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 8px;
  }
  table {
    font-size: 19px;
  }
  .highlight {
    background-color: #eff6ff;
    border-left: 5px solid #3b82f6;
    padding: 10px 15px;
    border-radius: 4px;
  }
---

# Human Activity Recognition: Complete ML Pipeline
### Comparative Study of Feature Selection, Dimensionality Reduction, and Ensembles on Smartphone Sensor Signals

**Course:** Practical Applications of AI & Machine Learning (PAIML Lab)  
**Students:** Kartikeya Jain & Ayush Yadav  
**Supervised By:** Dr. Pruthwik Mishra & Dr. Praveen Chandaliya  
**GitHub Repository:** `https://github.com/<username>/<repo>`

---

## Slide 1: Problem Statement

- **Objective:** Accurately classify physical human activities in real-time from tri-axial smartphone accelerometer and gyroscope signals.
- **The Core Challenges:**
  - **High Dimensionality:** 561 continuous variables extracted from raw time-series sensor windows.
  - **Extreme Multicollinearity:** Sensor channels measuring identical body axes carry redundant information.
  - **Stationary Posture Ambiguity:** Discerning fine-grained postural transitions between static states (`SITTING` vs. `STANDING`).
- **Target Constraint:** Design a production-grade machine learning system achieving **$\ge 95\%$ accuracy** with minimal latency and reduced feature footprint.

---

## Slide 2: Motivation

- **Digital Healthcare & Elderly Well-being:**
  - Automated fall detection, continuous activity monitoring, and rehabilitation assessment for patients with movement disorders.
- **Pervasive & Wearable Computing:**
  - Eliminates the need for expensive motion-capture laboratories; relies entirely on commodity smartphones.
- **Edge ML & Battery Optimization:**
  - Mobile processors have strict energy budgets. Compressing 561 features via **PCA** and **Sequential Feature Selection** directly reduces CPU cycles, RAM usage, and battery consumption on wearable devices.

---

## Slide 3: Dataset Overview

- **Source:** UCI Smartphone Human Activity Recognition (HAR) Dataset.
- **Subjects:** 30 human volunteers (aged 19–48) wearing waist-mounted smartphones (Samsung Galaxy S II) capturing sensor data at 50 Hz.
- **Scale & Structure:**
  - **Total Samples:** **10,299 instances** (Training: 7,352 | Test: 2,947).
  - **Feature Dimension:** **561 engineered features** (mean, std, energy, entropy, autoregression coefficients).
- **Target Classes (6 Balanced Activities):**
  - **Dynamic:** `WALKING` (1,722), `WALKING_UPSTAIRS` (1,544), `WALKING_DOWNSTAIRS` (1,406).
  - **Static:** `SITTING` (1,777), `STANDING` (1,906), `LAYING` (1,944).
- **Data Integrity:** **0 missing values**, **0 duplicate rows**, $<1\%$ outlier rate.

---

## Slide 4: Approaches (Part 1 — Preprocessing & Unsupervised Discovery)

- **Volunteer-Wise Grouping (`GroupKFold`):**
  - The same volunteer's movement style must not appear in both train and validation splits. Grouping by volunteer ID prevents **biometric identity leakage**.
- **Data Standardization:**
  - Applied `StandardScaler` fitted strictly on training data and mapped onto the test partition.
- **Unsupervised Structure Discovery (Without Class Labels):**
  - **KMeans++ ($k=2$):** Silhouette analysis peaked at $k=2$. The algorithm spontaneously separated **Dynamic activities** from **Static activities** with **99.7% alignment**.
  - **DBSCAN:** Density clustering on 10 PCA dimensions discovered **5 dense clusters** and isolated **371 boundary noise points** ($\text{ARI} = 0.320$).

---

## Slide 5: Approaches (Part 2 — Feature Selection & Dimensionality Reduction)

- **Sequential Feature Selection (SFS):**
  - **Phase 1 (Filter):** Dropped redundant pairs ($r > 0.95$) & selected top 30 features by ANOVA F-score.
  - **Phase 2 (Wrapper):** Executed greedy Forward and Backward SFS with 3-fold CV down to **15 features**.
  - *Outcome:* 11 features independently selected by both directions. Backward SFS + SVM achieved **87.51% accuracy** using just 15 features.
- **Dimensionality Reduction (PCA):**
  - Projecting onto orthogonal principal components.
  - **102 components** captured **95% of cumulative explained variance** (**81.8% compression**).
- **Supervised Benchmarking & Meta-Ensembles:**
  - Evaluated Logistic Regression, SVM (RBF), Random Forest, KNN, Decision Tree.
  - Combined top models into **Hard and Soft Voting Ensembles**.

---

## Slide 6: Complete Pipeline Architecture

```
[Raw Smartphone Sensor Window (561 Features)]
                     │
                     ▼
  [StandardScaler (Learned on Train Only)]
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
  [Full Features]  [PCA (95%)]  [SFS (Top 15)]
   (561 dims)      (102 dims)     (15 dims)
         │           │           │
         └───────────┼───────────┘
                     ▼
  [5 Model Benchmarks (LR, SVM, RF, KNN, DT)]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
 [5-Fold Group CV]     [Voting Ensembles (Hard & Soft)]
         │                       │
         └───────────┬───────────┘
                     ▼
 [Final Evaluation: 95.96% Acc | Leaderboard | Error Analysis]
```

---

## Slide 7: Experimental Results (Single Models & Compression)

| Feature Representation | Model | Features | Accuracy | Macro F1 | Training Time |
|---|---|:---:|:---:|:---:|:---:|
| **Full Feature Space** | Logistic Regression | 561 | **95.52%** | **0.9551** | 9.68s |
| Full Feature Space | Support Vector Machine (RBF) | 561 | 95.18% | 0.9511 | 2.43s |
| Full Feature Space | Random Forest (100 trees) | 561 | 92.67% | 0.9251 | 0.82s |
| Full Feature Space | K-Nearest Neighbors ($k=5$) | 561 | 88.36% | 0.8805 | 0.18s |
| Full Feature Space | Decision Tree | 561 | 86.22% | 0.8595 | 3.10s |
| **PCA (95% Variance)** | **SVM (RBF)** | **102** | **93.76%** | **0.9362** | **0.67s** |
| PCA (95% Variance) | Logistic Regression | 102 | 93.11% | 0.9298 | 6.32s |
| **Backward SFS** | **SVM (RBF)** | **15** | **87.51%** | **0.8731** | **0.50s** |

<div class="highlight">
<strong>Key Insight:</strong> PCA compressed features by 81.8% while preserving 98.5% of full-feature accuracy, reducing SVM training time by nearly 4x.
</div>

---

## Slide 8: Experimental Results (Ensembles & Leaderboard)

- **Ensemble Meta-Classifiers (LR + SVM + Random Forest):**
  - **Soft Voting (All Features):** **95.96% Accuracy**, **0.9588 Macro F1** (Rank 1 Overall).
  - **Hard Voting (All Features):** **95.93% Accuracy**, **0.9588 Macro F1**.
  - **Soft Voting on PCA (102 Features):** **93.99% Accuracy**, **0.9391 Macro F1**.
- **Consensus Behavior:**
  - Hard and Soft voting disagreed on only **39 out of 2,947 test rows** ($1.3\%$).
- **5-Fold Volunteer Cross-Validation (Generalizability):**
  - Logistic Regression (PCA Pipeline): $91.89\% \pm 4.29\%$
  - SVM (PCA Pipeline): $90.92\% \pm 4.40\%$
  - Confirms models generalize cleanly across previously unseen individuals.

---

## Slide 9: Challenges Faced & Diagnostic Insights

1. **Multicollinearity:**
   - 23.3% of feature pairs had $r > 0.70$. Standard algorithms risked severe coefficient variance; solved via ANOVA pruning and orthogonal PCA projection.
2. **Biometric Leakage Mitigation:**
   - Standard random shuffling artificially inflates validation accuracy because subjects appear in both splits. Mitigated by using **`GroupKFold`** on subject ID.
3. **The "Sitting vs. Standing" Dilemma:**
   - **53.8% of all model errors (71 of 132 misclassifications)** were confusions between `SITTING` and `STANDING`.
   - *Physical Cause:* Both activities represent static postures with identical gravitational acceleration ($1g$ downward); only subtle hip/thigh angles differ.

---

## Slide 10: Major Learnings from the Project

- **The Curse of Dimensionality in Clustering:**
  - High-dimensional distance metrics collapse in 561-D space. Running DBSCAN on 10 PCA dimensions restored density contrast and isolated meaningful noise.
- **Efficacy of Compression for Edge Computing:**
  - 102 PCA components provide near-identical accuracy ($93.76\%$) at a fraction of compute cost, rendering the model viable for low-power microcontrollers.
- **Complementary Ensemble Strengths:**
  - Averaging probability distributions (Soft Voting) between a linear model, a maximum-margin kernel model, and an ensemble of decision trees effectively smooths decision boundaries.

---

## Slide 11: Summary & Key Deliverables

- **End-to-End Pipeline Completed:**
  - Ingestion $\rightarrow$ EDA $\rightarrow$ Unsupervised Analysis $\rightarrow$ SFS $\rightarrow$ PCA $\rightarrow$ Modeling $\rightarrow$ Ensembles $\rightarrow$ Diagnostics.
- **Top Classifier:** **Soft Voting Ensemble (95.96% Accuracy | 0.9588 Macro F1)**.
- **Top Edge-Optimized Model:** **SVM on 102 PCA Components (93.76% Accuracy | 0.67s train time)**.
- **Deliverables:**
  - Clean, fully reproducible Jupyter Notebook: `HAR_ML_Pipeline.ipynb`
  - GitHub Repository Documentation: `README.md`
  - Presentation Deck: `Project_Presentation.md` & `presentation.html`

---

## Slide 12: Questions & Discussion

### Thank You!

**Kartikeya Jain & Ayush Yadav**  
B.Tech — Practical Applications of AI & Machine Learning  

**Faculty Mentors:**  
Dr. Pruthwik Mishra & Dr. Praveen Chandaliya  

- **GitHub Repository:** `https://github.com/<username>/<repo>`
- **Notebook & Results:** `HAR_ML_Pipeline.ipynb`
