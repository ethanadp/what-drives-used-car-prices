# Used Car Price Modeling and Feature Analysis

This project applies the CRISP-DM framework to a large-scale used car dataset to build a predictive model of vehicle prices and uncover the most influential factors that affect valuation. The final output is designed to help used car dealerships better understand pricing drivers and fine-tune their inventory strategy.

---

## Business Objective

Used car dealers want to better understand which features most influence vehicle prices, and how to price their inventory more effectively. The goal of this project was to build a regression model that not only predicts price accurately, but also provides interpretable insights into what drives those prices.

---

## Dataset

- Source: Kaggle (subset of original 3M+ vehicle records)
- Records: ~426,000 listings
- Key Features: `price`, `year`, `manufacturer`, `model`, `condition`, `odometer`, `state`, etc.

---

## Process Overview (CRISP-DM)

### 1. Business Understanding
Defined the dealership's need to optimize pricing using historical listing data.

### 2. Data Understanding
Performed:
- Descriptive stats and visualizations
- Data quality checks
- Outlier filtering (e.g., price > $100,000 removed)

### 3. Data Preparation
- Feature selection based on business relevance
- Created `model_avg_price` using K-Fold Target Encoding with smoothing
- One-hot encoded remaining categorical variables
- Finalized numeric/categorical split for pipeline

### 4. Modeling
Tested:
- **Linear Regression**
- **Ridge Regression** (Best Performer)
- **LASSO**
- **ElasticNet**

**Final Model:**
- Ridge Regression  
- R² = 0.65  
- RMSE ≈ $7,880  
- MAE ≈ $5,020

### 5. Evaluation
- Residuals centered around zero; variance increases for high-end cars
- Manufacturer explained ~59% of model’s predictive power
- Top features included `manufacturer_ferrari`, `condition_new`, and `model_avg_price`

### 6. Deployment
Prepared a report with:
- Performance metrics
- Visualized feature importance (individual + grouped)
- Strategic recommendations for dealership pricing and inventory selection

---

## Key Insights

- **Manufacturer** is the most powerful pricing driver.
- **Vehicle condition** and **historical model pricing** (via `model_avg_price`) are strong contributors.
- **Regional variability** (state) affects price — important for geographically distributed dealers.
- Grouped importance visualization helped translate the model into actionable business strategies.

---

## Visual Highlights

- Top 20 most influential features (Ridge coefficients)
- Grouped feature importance as % of total model weight
- Residual distribution plot
- Average price matrix by `manufacturer` × `condition`
- Price variance across states per manufacturer

---

## Files

- `car_price_modeling.ipynb` — Main notebook with full analysis
- `README.md` — Project summary
- `plots/` — Optional folder for generated visuals

---

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, seaborn, matplotlib

To run locally:
```bash
pip install -r requirements.txt
