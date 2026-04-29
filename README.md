

---

# Cancer Detection and Gene Analysis using AI

**Live App:** [https://ranjith445-histopath-multimodal-biomarker.hf.space/](https://ranjith445-histopath-multimodal-biomarker.hf.space/)

---

## What is this project about?

Doctors study cancer in two main ways:

1. **Looking at tissue images** under a microscope
2. **Analyzing genes** to understand risk

Usually, these are done separately. Also, many AI systems give results but **do not explain how they reached them**.

This project builds an AI system that:

* Looks at tissue images
* Studies gene data
* Combines both
* Explains its decisions clearly

---

## What does this system do?

The system can:

* Check if cancer is present in tissue images
* Predict whether a patient is **high risk or low risk** based on genes
* Identify **important genes related to cancer**
* Show **which part of the image was important**
* Generate a **complete report**

---

## How does it work (simple explanation)

The system works in two parts and then combines them.

---

### Part 1: Image Analysis

* The AI looks at small tissue images
* It learns patterns that indicate cancer
* It predicts:

  * Tumor present
  * Tumor not present

It also highlights:

* The exact regions it focused on

---

### Part 2: Gene Analysis

* The system looks at gene data from patients
* It studies around **100 important cancer-related genes**
* It predicts:

  * High risk
  * Low risk

It also identifies:

* Which genes influenced the decision the most

---

### Final Step: Combine Both

* Image result + gene result are combined
* A final report is created
* Shows both visual and genetic insights

---

## What makes this project special?

### 1. Combines two types of data

* Images (what the tissue looks like)
* Genes (what is happening inside cells)

---

### 2. Provides explanations

Instead of just giving results, it shows:

* Image heatmap → where it looked
* Gene importance → which genes matter

---

### 3. Generates reports

* Creates a downloadable PDF
* Includes predictions and explanations

---

## What data was used?

### Image Data

* Dataset: PatchCamelyon (PCam)
* Contains small tissue image patches

### Gene Data

* 50 patients
* 100 cancer-related genes

---

## What models are used?

The system uses two AI models:

1. **Image Model (EfficientNet)**

   * Detects cancer in images

2. **Gene Model (Random Forest)**

   * Predicts risk using gene data

---

## What results does it give?

* Image model accuracy: 90%
* Gene model accuracy: 68%

It also identifies important genes like:

* BRCA1
* TP53
* PIK3CA

---

## Features of the application

* Select patient data
* View tissue images
* Get cancer prediction
* View gene-based risk prediction
* See highlighted image regions
* See important genes
* Download full report

---

## Important Note

* Uses a **limited number of patients (50)**
* Built for **learning and demonstration**

This is not a medical tool.

---

## Limitations

* Small dataset
* Not tested in real hospitals
* Needs more real-world data

---

## Disclaimer

This project is for educational and research purposes only.
Do not use it for medical diagnosis.

---

## One-Line Summary

An AI system that combines tissue images and gene data to detect cancer and explain its decisions.

---

## Author

Built by M. Ranjith Kumar as a biomedical AI portfolio project.

---

