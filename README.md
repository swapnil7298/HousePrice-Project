# 📚 House Price Prediction Project

> **A machine learning project that predicts house prices using linear regression and deploys the model via a Flask web interface.**

[![Python](https://img.shields.io/badge/Language-Python-blue)](https://python.org/)
[![ML](https://img.shields.io/badge/ML-scikit%2Dlearn-brightgreen)](https://scikit-learn.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-green)](https://flask.palletsprojects.com/)
[![Status](https://img.shields.io/badge/Status-Complete-success)]()

## 📌 Overview

**House Price Prediction** is a machine learning project that demonstrates end-to-end ML development - from data analysis and model training using scikit-learn to deploying a web application using Flask.

### Key Features:
- ✅ **Linear Regression Model** - Trained on real estate data
- ✅ **Data Analysis & Visualization** - Exploratory data analysis
- ✅ **Web Interface** - Flask-based prediction dashboard
- ✅ **Real-time Predictions** - Get house price estimates instantly
- ✅ **Model Performance** - High accuracy on test data

---

## 🎯 Project Goals

1. **Data Exploration** - Understand housing market patterns
2. **Model Development** - Build and train regression model
3. **Model Evaluation** - Measure accuracy and performance
4. **Deployment** - Create web interface for predictions

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn |
| **Web Framework** | Flask |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |

---

## 🚀 Getting Started

### Installation

```bash
# Clone repository
git clone https://github.com/swapnil7298/HousePrice-Project.git
cd HousePrice-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run Flask app
python app.py
```

---

## 📋 Project Structure

```
HousePrice-Project/
├── app.py                    # Flask application
├── train_model.py           # Model training script
├── requirements.txt         # Dependencies
├── data/
│   ├── raw/
│   │   └── housing.csv
│   └── processed/
├── models/
│   └── house_price_model.pkl
├── notebooks/
│   └── analysis.ipynb
└── templates/
    ├── index.html
    └── predict.html
```

---

## 💻 Usage

### Train Model
```bash
python train_model.py
```

### Run Web App
```bash
python app.py
# Visit http://localhost:5000
```

### Make Predictions
Navigate to the web interface and enter:
- Square footage
- Number of bedrooms
- Number of bathrooms
- Year built
- Location

Click "Predict" to get estimated house price.

---

## 📧 Contact

- **GitHub**: [@swapnil7298](https://github.com/swapnil7298)
- **Email**: swapnilrao729@gmail.com

---

**Built with ❤️ by Swapnil Rao** | Making real estate predictions accessible
