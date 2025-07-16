# 🛍️ Online Retail Sales Forecasting

This project uses historical online retail transaction data to forecast future daily sales using machine learning. The goal is to predict daily revenue trends to aid in inventory planning, marketing strategies, and business decision-making.

## 📁 Dataset
The dataset contains transactional records with the following key columns:
- `InvoiceNo`: Unique transaction ID (cancellations start with 'C')
- `InvoiceDate`: Timestamp of the transaction
- `Quantity`: Units sold per product
- `UnitPrice`: Price per unit in GBP
- `Country`: Customer location

---

## 🔍 Problem Statement
Forecast total daily sales based on past patterns using time-series features and machine learning models such as **Random Forest** and **XGBoost**.

---

## 🧪 Workflow

### 1. **Data Cleaning**
- Removed cancelled transactions (those with InvoiceNo starting with 'C')
- Converted `InvoiceDate` to datetime with `dayfirst=True`
- Calculated `TotalSales = Quantity × UnitPrice`
- Aggregated sales at daily granularity

### 2. **Feature Engineering**
- Lag features: previous 30 days of sales
- Time features: day of week, month, is weekend, week of year
- Outlier treatment using 99th percentile capping

### 3. **Modeling**
- **XGBoost Regressor** used to model sales
- Data split: 80% train, 20% test (without shuffling to preserve time)
- Evaluation metrics:
  - **RMSE:** 14,960
  - **R² Score:** 0.46

### 4. **Visualization**
- Line plot comparing actual vs. predicted daily sales

---

## 📈 Results

The final model captures weekly sales patterns and general trends but slightly underpredicts peak days. It shows strong potential for practical forecasting and can be improved further using advanced time-series models or hybrid approaches.

---

## 🔧 Future Improvements
- Add rolling mean and std features
- Try log transformation of target
- Use Facebook Prophet or ARIMA
- Explore hybrid classification + regression modeling

---

## 📦 Requirements

See `requirements.txt` below for the environment dependencies.

---

## 👨‍💻 Author

**Mohaemn Saber**  
📍 Business Information Systems, Helwan University  
📊 Focused on Data Analytics & Forecasting  
🗓️ Project Date: July 2025

