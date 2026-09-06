# 🚗 Car Price Prediction Web App

A Machine Learning-powered web application that predicts the **resale price of used cars** based on user inputs such as fuel type, transmission, kilometers driven, and more.

---

## 🌐 Live Demo

👉 [https://car-price-prediction-1-wjcu.onrender.com](https://car-price-prediction-1-wjcu.onrender.com)

---

## 🎬 Demo Preview

![Demo](static/images/demo.gif)

---

## ✨ Features

- Instant used car price prediction in Indian Rupees (Lakhs)
- Clean, animated frontend UI
- Random Forest ML model with **R² Score: 0.93**
- Flask REST backend with a `/predict` POST endpoint
- Config-driven model loading via `config.yaml`
- Production-ready with `gunicorn` and a `Procfile`
- Deployable on Render

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3 |
| Web Framework | Flask |
| ML Library | Scikit-learn 1.6.1 |
| Data Processing | NumPy, Pandas |
| Serving | Gunicorn |
| Config | PyYAML |
| Frontend | HTML / CSS |
| Deployment | Render |

---

## 🧠 ML Pipeline

1. **Data Cleaning** — Handle nulls and outliers
2. **Feature Engineering** — Car age derived from year; relevant features selected
3. **Encoding** — Manual one-hot encoding for `fuel_type`, `seller_type`, `transmission`
4. **Scaling** — StandardScaler applied to all 8 input features
5. **Model Training** — Random Forest Regressor
6. **Serialization** — Model and scaler saved as `.pkl` files
7. **Deployment** — Served via Flask + Gunicorn

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| R² Score | 0.93 |
| MAE | 0.45 Lakhs |

---

## 🔢 Input Features

| Feature | Type | Description |
|---|---|---|
| `present_price` | Float | Current ex-showroom price (₹ Lakhs) |
| `kms_driven` | Float | Total kilometers driven |
| `owner` | Integer | Number of previous owners (0, 1, 2, 3) |
| `car_age` | Integer | Age of the car in years |
| `fuel_type` | Categorical | Petrol / Diesel / CNG |
| `seller_type` | Categorical | Dealer / Individual |
| `transmission` | Categorical | Manual / Automatic |

---

## ⚙️ Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Arjunkalliyadath/car-price-prediction.git
cd car-price-prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

Then open [http://localhost:10000](http://localhost:10000) in your browser.

---

## 🚀 Deploy to Render

This app is production-ready out of the box:

- **`Procfile`** — tells the platform to serve with `gunicorn app:app`
- **Port** — dynamically read from `PORT` environment variable

---

## 📂 Project Structure

```
car-price-prediction/
│
├── static/
│   └── images/
│       └── demo.gif
├── templates/
│   └── index.html
├── app.py                  # Flask app & prediction logic
├── config.yaml             # Model and scaler paths
├── car_price_model.pkl     # Trained Random Forest model
├── scaler.pkl              # Fitted StandardScaler
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn entry point for deployment
└── README.md
```

---

## 🔧 Configuration

Model and scaler paths are managed via `config.yaml`, making it easy to swap models without touching the code:

```yaml
model_path: car_price_model.pkl
scaler_path: scaler.pkl
```

---

## 📬 Contact

**Arjun Kalliyadath**
- GitHub: [@Arjunkalliyadath](https://github.com/Arjunkalliyadath)
- Email: arjunkalliyadath2001@gmail.com
