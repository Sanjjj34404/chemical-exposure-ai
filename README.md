# 🔬 Explainable AI for Chemical Exposure Outcome Prediction

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3.5-green)](https://lightgbm.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> An interpretable machine learning framework for predicting health outcomes and recommending treatments for workplace chemical exposures, featuring multi-lingual explanations powered by LLMs.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 Overview

This project develops an **explainable AI system** for predicting health outcomes and treatment recommendations following workplace chemical exposures. Unlike traditional black-box models, our system provides **human-readable explanations** in multiple languages, making AI predictions accessible to workers, healthcare providers, and safety officers.

### The Problem

Workplace chemical exposures pose significant health risks across industries like construction, manufacturing, and healthcare. Timely and accurate prediction of health outcomes and appropriate treatment recommendations can:
- ✅ Improve patient outcomes through early intervention
- ✅ Optimize resource allocation in occupational health settings
- ✅ Reduce long-term health complications
- ✅ Enhance workplace safety protocols

### Our Solution

A **dual-model explainable AI system** that:
1. **Predicts health outcomes** (Full Recovery, Partial Recovery, Ongoing Treatment)
2. **Recommends treatments** (Medical Attention, Hospitalization, Observation, etc.)
3. **Explains predictions** in simple language using LIME + LLM integration
4. **Supports multiple languages** (English, Hindi, Tamil)

---

## ✨ Key Features

### 🤖 Advanced Machine Learning
- **LightGBM models** achieving 92.34% accuracy for outcome prediction
- **Multi-class classification** for 3 outcomes and 6 treatment categories
- **Feature engineering** including severity indices and interaction terms
- **Robust handling** of categorical and numerical features

### 🔍 Explainability & Transparency
- **LIME (Local Interpretable Model-agnostic Explanations)** for feature importance
- **Groq LLM integration** for natural language explanation generation
- **Worker-friendly** explanations tailored to non-technical users
- **Readability metrics** demonstrating explanation quality

### 🌍 Accessibility
- **Multi-lingual support**: English, Hindi (हिंदी), Tamil (தமிழ்)
- **Interactive web interface** built with Streamlit
- **Top-3 treatment recommendations** with confidence scores
- **Visual explanations** with feature importance charts

### 📊 Comprehensive Analysis
- **ROC-AUC analysis** with all classes >0.91
- **Precision-Recall curves** for imbalanced class handling
- **Cross-validation** with 5-fold stratified splits
- **Baseline comparisons** against Random Forest, Logistic Regression, Naive Bayes

---

## 🎬 Demo

### Prediction Interface
![Prediction Interface](docs/images/demo_interface.png)

### Explanation Example
![Explanation Example](docs/images/demo_explanation.png)

### Multi-lingual Support
![Multi-lingual](docs/images/demo_multilingual.png)

> **Note:** Add screenshots to `docs/images/` folder

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment

### Step 1: Clone the Repository
bash
git clone https://github.com/yourusername/chemical-exposure-ai.git
cd chemical-exposure-ai

### Step 2: Create Virtual Environment (Recommended)
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Set Up Environment Variables
GROQ_API_KEY=your_groq_api_key_here
Get your free Groq API key from console.groq.com

## 📁 Project Structure
chemical-exposure-ai/
│
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .env.example                       # Environment variables template
├── 📄 LICENSE                            # MIT License
│
├── 📊 exposure_data.csv                  # Dataset (10,000 samples)
│
├── 🐍 model_train.py                     # Model training script
├── 🐍 explain.py                         # Streamlit web application
│
├── 📂 models/                            # Saved models & encoders
│   ├── outcome_lgb.txt                   # Outcome prediction model
│   ├── treatment_lgb.txt                 # Treatment recommendation model
│   ├── outcome_target_encoder.pkl        # Outcome label encoder
│   ├── treatment_target_encoder.pkl      # Treatment label encoder
│   ├── encoders.pkl                      # Feature encoders
│   └── lime_background.pkl               # LIME background data

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.



