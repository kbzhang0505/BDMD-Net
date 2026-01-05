<div align="center">

# How Local Details Meet Global Structures  
## A Parallel Multi-Scale Dual-Branch Network for Low-Light Image Enhancement

</div>

---

<div align="center">
  <img src="Figure/Fig1.png" alt="Overview of BDMD-Net" width="90%" />
</div>

---

## 🔑 Core Contributions

- **BDMD-Net**:  
  We propose a novel **Brightness–Detail coordinated Multi-scale Dual-branch Network (BDMD-Net)** for low-light image enhancement.  
  The framework leverages **Transformers** to model global brightness distribution and **CNNs** to capture fine-grained local details.

- **Cross-scale Brightness-Guided Attention (CBGA)**:  
  A cross-scale attention mechanism is introduced to enhance interactions among different feature scales by explicitly incorporating **brightness priors** derived from low-light inputs.

- **Frequency-Domain Phase–Amplitude Separated Fusion**:  
  We design an effective feature fusion strategy that separates **phase** and **amplitude** components in the frequency domain.  
  This strategy independently models **global structural information** and **local texture details**, effectively avoiding feature aliasing while preserving fine details.

---

## 🧩 Core Module Description

<div align="center">
  <img src="Figure/Fig2.png" alt="Core modules of BDMD-Net" width="90%" />
</div>

**Module illustration:**

- **(a)** Brightness prior learning module  
- **(b)** CBGA (Case 2)  
- **(c)** CBGA (Case 3)  
- **(d)** APSF: Amplitude–Phase Separated Fusion module  

---

## 📊 Experimental Results

<div align="center">
  <img src="Figure/Fig3.png" alt="Results on LOL datasets" width="90%" />
</div>

<div align="center">
  <img src="Figure/Fig4.png" alt="Results on SMID dataset" width="90%" />
</div>

**Visual comparisons** of different approaches on  
**LOLv1**, **LOLv2-real**, **LOLv2-synthetic**, and **SMID** datasets.

---

<div align="center">
  <img src="Figure/Fig5.png" alt="Comparison with MIR" width="90%" />
</div>

**Visual comparison between BDMD-Net and MIR** across four benchmark datasets.

---

## 📌 Notes

- This repository provides the implementation of **BDMD-Net**.
- Please refer to the paper for more architectural and theoretical details.

---

<div align="center">

✨ *If you find this work useful, please consider citing our paper.* ✨

</div>
