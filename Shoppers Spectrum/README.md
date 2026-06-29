# 🛍️ Shoppers Spectrum Dashboard

An interactive, end-to-end Machine Learning web application built with Streamlit. The dashboard helps retail businesses drive engagement and revenue through dual-engine analytical tracks: **Product Recommendation (Item-Based Collaborative Filtering)** and **Customer Segmentation (RFM + K-Means Clustering)**.

---

## 🚀 Key Modules & Features

### 1. 🛒 Product Recommendation Module
* **Goal:** Increase cross-selling opportunities by identifying items frequently bought together.
* **Mechanism:** Uses a precalculated `cosine_similarity` matrix across thousands of unique inventory item buying fingerprints.
* **UI Features:**
  * Auto-complete dropdown search matching unique item descriptions.
  * Side-by-side execution action blocks with a dynamic state-flushing **Clear Selection** tool.
  * Modern, interactive styled item recommendation cards.

### 2. 📊 Customer Segmentation Module
* **Goal:** Classify client accounts into target profiles to optimize marketing campaigns.
* **Mechanism:** Applies an `np.log1p` transformation pipeline followed by a trained `K-Means` model based on real-time **Recency**, **Frequency**, and **Monetary (RFM)** values.
* **UI Features:**
  * Parameter text configuration using container form blocks to eliminate UI shifting.
  * Dynamic, theme-isolated, high-contrast visual result cards mapped to target loyalty profiles:
    * 💰 **High-Value Customer** (Active, high spending, high frequency)
    * 🔹 **Regular Customer** (Steady interaction, standard volume)
    * ⏱️ **Occasional Customer** (Infrequent but active buyer)
    * ⚠️ **Customer At-Risk** (Stale timeline, low transaction history)

---

## 🛠️ Project File Architecture

Ensure your directory contains the following assets for the dashboard to launch successfully:

```text
Shoppers Spectrum/
│
├── EDA Charts/             # Generated directory containing visualization plots
│
├── Models/                 # Serialized Machine Learning artifacts (Capital M)
│   ├── item_similarity.pkl # Cosine similarity lookup matrix DataFrame
│   ├── scaler.pkl          # Trained StandardScaler instance
│   └── kmeans_model.pkl    # Configured K-Means clustering model instance
│
├── Shoppers Spectrum.ipynb # Core notebook for data processing & model pipeline
├── app.py                  # Main Streamlit dashboard interface application file
├── cleaned.csv             # Preprocessed transaction data used by the models
└── requirements.txt        # Python dependency packages file