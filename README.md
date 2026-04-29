Here is your **final cleaned README**, with your original content preserved and the **PDF explanation made structured (not scattered)** by improving the *Final Step* section only.

---

# Cancer Detection and Gene Analysis using AI

**Live App:** [https://ranjith445-histopath-multimodal-biomarker.hf.space/](https://ranjith445-histopath-multimodal-biomarker.hf.space/)

---

## What is this project about?

Doctors study cancer in two main ways:

Looking at tissue images under a microscope
Analyzing genes to understand risk

Usually, these are done separately. Also, many AI systems give results but do not explain how they reached them.

This project builds an AI system that:

Looks at tissue images
Studies gene data
Combines both
Explains its decisions clearly

---

## What does this system do?

The system can:

Check if cancer is present in tissue images
Predict whether a patient is high risk or low risk based on genes
Identify important genes related to cancer
Show which part of the image was important
Generate a complete report

👉 The generated report includes:
https://drive.google.com/file/d/1VJSQlll3CBqnvtbT1hJy3fGLs2wcSuWB/view?usp=sharing

* Original tissue image and highlighted regions (heatmap) showing where the AI focused
* Tumor prediction with confidence score
* Gene-based risk prediction (high or low risk)
* List of important genes and their influence on risk
* Final combined interpretation of image and gene analysis

---

## How does it work (simple explanation)

The system works in two parts and then combines them.

### Part 1: Image Analysis

The AI looks at small tissue images

It learns patterns that indicate cancer

It predicts:

Tumor present
Tumor not present

It also highlights:

The exact regions it focused on

👉 In the report:

* A colored heatmap (Grad-CAM) is shown

* Colors represent how important each region is for the prediction:

  Red / Yellow → Highly important areas (strong influence on decision)
  Orange → Moderately important
  Blue → Less important or ignored regions

* Highlighted regions indicate where the model focused to detect tumor patterns

---

### Part 2: Gene Analysis

The system looks at gene data from patients

It studies around 100 important cancer-related genes

It predicts:

High risk
Low risk

It also identifies:

Which genes influenced the decision the most

👉 In the report:

* Each gene is shown with its contribution
* “Decreases risk” means safer
* “Increases risk” means higher concern
* SHAP values indicate how strongly each gene influenced the prediction

---

### Final Step: Combine Both

Image result + gene result are combined
A final report is created
Shows both visual and genetic insights

👉 How to read the report step-by-step:

1. First, look at the image:

   * The heatmap shows where the AI focused
   * Red/Yellow = most important areas
   * Blue = less important

2. Next, check the image prediction:

   * Shows whether tumor patterns are present
   * Includes confidence score

3. Then, look at gene analysis:

   * Shows High risk or Low risk
   * Lists important genes
   * “Decreases risk” = safer
   * “Increases risk” = higher concern

4. Finally, read the combined result:

   * Image + gene analysis together decide the final outcome

Example:

* Image shows tumor patterns
* Gene analysis shows low risk

Final meaning:
The condition may exist but is not aggressive

---

## What makes this project special?

1. Combines two types of data
   Images (what the tissue looks like)
   Genes (what is happening inside cells)

2. Provides explanations
   Instead of just giving results, it shows:

Image heatmap → where it looked
Gene importance → which genes matter

3. Generates reports
   Creates a downloadable PDF
   Includes predictions and explanations that are easy to understand

---

## What data was used?

### Image Data

Dataset: PatchCamelyon (PCam)
Contains small tissue image patches

### Gene Data

50 patients
100 cancer-related genes

---

## What models are used?

The system uses two AI models:

Image Model (EfficientNet)
Detects cancer in images

Gene Model (Random Forest)
Predicts risk using gene data

---

## What results does it give?

Image model accuracy: 90%
Gene model accuracy: 68%

It also identifies important genes like:

BRCA1
TP53
PIK3CA

---

## Features of the application

Select patient data
View tissue images
Get cancer prediction
View gene-based risk prediction
See highlighted image regions
See important genes
Download full report

---

## Important Note

Uses a limited number of patients (50)
Built for learning and demonstration
This is not a medical tool.

---

## Limitations

Small dataset
Not tested in real hospitals
Needs more real-world data

---

## Disclaimer

This project is for educational and research purposes only. Do not use it for medical diagnosis.

---

## One-Line Summary

An AI system that combines tissue images and gene data to detect cancer and explain its decisions.

---

## Author

Built by M. Ranjith Kumar as a biomedical AI portfolio project.

---

