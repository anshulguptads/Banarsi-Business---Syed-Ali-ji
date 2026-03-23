# Banarasi Traditional Products — Data-Driven Analytics Dashboard

A comprehensive Streamlit analytics platform for a startup bringing authentic Banarasi (Varanasi) traditional products to the pan-India market. The dashboard applies **classification, clustering, association rule mining, and regression** on consumer survey data to enable data-driven decision making across customer targeting, product bundling, pricing strategy, and marketing.

---

## Business Context

Banaras (Varanasi) is home to centuries-old artisanal crafts — from silk sarees and handwoven carpets to brass artware, Gulabi Meenakari, and premium paan. This project evaluates the market opportunity for scaling these products pan-India using a D2C and omnichannel approach.

A **28-question consumer survey** was designed to capture demographics, psychographics, product preferences, spending behavior, purchase barriers, and intent. A synthetic dataset of **2,000 respondents** was generated with realistic Indian demographic distributions, persona-driven conditional logic, noise injection, and deliberate outliers — mimicking real-world data collection.

### Five Products Under Analysis

| Product | Heritage | Market Signal |
|---|---|---|
| Banarasi Silk Sarees & Textiles | GI-certified since 2009 | Indian ethnic wear market: $197B (2024) → $558B (2033) |
| Handwoven Carpets & Rugs | Bhadohi-Mirzapur belt | India = 40% of global handmade carpet exports |
| Brass & Copper Artware | UNESCO-recognized Thatheri Bazaar | Home décor market: $26.9B (2025) → $42.4B (2034) |
| Premium Banarasi Paan | GI-certified since 2023 | Paan café franchises growing at 25% annually |
| Gulabi Meenakari Craft | GI-certified, found only in Varanasi | Rare craft with high gifting and export potential |

---

## Dashboard Pages

### Page 1 — Executive Summary
KPI cards showing total addressable market, interested customer count, average spend, top product, and biggest conversion barrier.

### Page 2 — Descriptive Analytics
Four interactive tabs covering demographic distributions (age × gender, city tier × income), product demand heatmaps by state and city tier, spending analysis by income and cultural affinity, and channel/barrier frequency charts.

### Page 3 — Customer Segmentation (Clustering)
- **Algorithm:** K-Means with elbow method and silhouette score for optimal K selection
- **Visualization:** PCA 2D scatter plot of customer segments
- **Output:** Expandable cluster profile cards (spend, intent, demographics, top products), per-cluster discount and marketing strategy table, and a CLV × Conversion priority matrix

### Page 4 — Association Rule Mining
- **Algorithm:** Apriori (mlxtend) with support, confidence, and lift metrics
- **Three analysis levels:**
  - Cross-product rules (which products are bought together)
  - Within-product rules (saree type × color, paan flavor combos)
  - Barrier → Trust builder rules (what reassurances address which concerns)
- **Visualization:** Interactive support × confidence scatter sized by lift

### Page 5 — Predictive Models
- **Classification (Purchase Intent):**
  - Models: Random Forest, XGBoost, Logistic Regression
  - SMOTE applied for class imbalance
  - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - Visuals: ROC curves, confusion matrix, top 15 feature importance
- **Regression (Annual Spending):**
  - Models: Linear Regression, Random Forest, Gradient Boosting
  - Log-transform on target for right-skewed spending data
  - Metrics: R², MAE, RMSE
  - Visuals: Predicted vs actual scatter, feature importance

### Page 6 — Prescriptive Actions
Translates analytical outputs into business strategy:
- Named product bundle recommendations derived from association rules
- Segment-specific marketing playbooks (geography, channel, lead product, CLV per segment)
- Phased launch priority sequence ranked by CLV × conversion score

### Page 7 — New Customer Scorer
- Upload a CSV of new survey respondents
- Trained models score each customer with: purchase probability (%), predicted annual spend (₹), cluster assignment, and a composite priority score
- Download scored results as CSV for CRM integration
- Includes a sample test file generator for validation

---

## Tech Stack

| Component | Technology |
|---|---|
| UI Framework | Streamlit |
| ML Models | Scikit-learn, XGBoost |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Association Mining | mlxtend (Apriori) |
| Visualization | Plotly |
| Data Processing | Pandas, NumPy |

---

## Dataset Overview

| Attribute | Value |
|---|---|
| Respondents | 2,000 |
| Columns | 119 (14 categorical + 102 binary + 1 numeric + 2 metadata) |
| Survey Questions | 28 across 7 sections |
| Classification Target | Q28 — Purchase Likelihood (72/28 split) |
| Regression Target | Q24 — Annual Spending (mean ₹47,639, median ₹15,400, skewness 3.37) |
| Personas | 6 latent archetypes driving conditional response generation |
| Noise | 5–8% response inconsistencies |
| Outliers | 2–3% deliberate anomalies (58 spending outliers, 14 student high-spenders) |
| State Coverage | 34 states/UTs |

---

## Local Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/banarasi-analytics-dashboard.git
cd banarasi-analytics-dashboard

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## Streamlit Cloud Deployment

1. Push all three files (`app.py`, `requirements.txt`, `banarasi_products_survey_2000.csv`) to the **root** of a GitHub repository — no subfolders needed.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect your GitHub account.
3. Select the repository, set **Main file path** to `app.py`.
4. Click **Deploy**. Streamlit Cloud reads `requirements.txt` automatically.

---

## Repository Structure

```
├── app.py                                  # Streamlit dashboard (all 7 pages)
├── requirements.txt                        # Python dependencies with pinned versions
├── banarasi_products_survey_2000.csv       # Synthetic survey dataset (2,000 respondents)
└── README.md                               # This file
```

---

## Analytics Framework

The dashboard follows a four-layer analytics progression:

```
Descriptive  →  "What is happening?"     →  KPIs, distributions, heatmaps
Diagnostic   →  "Why is it happening?"   →  Clustering, association mining
Predictive   →  "What will happen?"      →  Classification, regression
Prescriptive →  "What should we do?"     →  Bundles, pricing, launch roadmap
```

Each layer feeds into the next — descriptive insights inform clustering features, cluster profiles feed into predictive model interpretation, and predictive scores drive prescriptive recommendations.

---

## Key Business Decisions This Dashboard Answers

| Decision | Analysis Used |
|---|---|
| Which customer segment to target first? | Clustering + Priority Matrix |
| What products to bundle together? | Association Rule Mining |
| How to price and discount per segment? | Clustering + Regression |
| Where to launch geographically? | Descriptive (state heatmaps) + Clustering |
| Which marketing channel for each segment? | Descriptive (channel analysis) + Clustering |
| Will a new lead convert? | Classification (Random Forest / XGBoost) |
| How much will a customer spend annually? | Regression (Gradient Boosting) |
| What barriers to address on the website? | Association Mining (Barrier → Trust rules) |

---

## Survey Design

The 28-question survey (V2) covers six research dimensions:

1. **Demographics** (Q1–Q6): Age, gender, state, city tier, occupation, income
2. **Household Context** (Q7): Family structure
3. **Psychographics** (Q8–Q10): Cultural affinity, Banarasi familiarity, sustainability values
4. **Shopping Behavior** (Q11–Q15): Frequency, occasions, channels, purchase factors, current brands
5. **Product Preferences** (Q16–Q22): Interest across 5 products with sub-preferences (saree types/colors, carpet types, brass items, paan flavors, meenakari items)
6. **Budget, Barriers & Intent** (Q23–Q28): Spending, premium willingness, purchase barriers, trust builders, purchase likelihood

---

## License

This project is for educational and business planning purposes. The dataset is synthetic and does not contain real personal information.

---

## Author

Built as a data-driven decision engine for evaluating the market opportunity of bringing authentic Banarasi traditional products to the pan-India consumer market.
