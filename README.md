# 📈 AI-Powered Trade Monitoring & Anomaly Detection System

An intelligent real-time stock market surveillance platform that combines ensemble anomaly detection, LSTM Autoencoders, market news analysis, and LLM-powered explanations to identify unusual trading behavior and notify users through Telegram.

The system continuously monitors live stock prices, detects anomalies using multiple machine learning models, enriches alerts with relevant news context, generates natural language explanations using Llama 3.1, and provides historical monitoring through a Streamlit dashboard.

---

## 🚀 Features

### Real-Time Market Monitoring
- Live stock price monitoring using Yahoo Finance
- Supports S&P 500 and NIFTY 50 stocks
- Continuous monitoring with configurable intervals

### Ensemble Anomaly Detection
Combines multiple anomaly detection techniques:

- Z-Score Detection
- Isolation Forest
- Local Outlier Factor (LOF)
- One-Class SVM
- Residual Error Detection
- LSTM Autoencoder

An anomaly is triggered only when multiple detectors agree, reducing false positives.

### AI-Powered Explanations
- Retrieves relevant market news using Serper API
- Uses Groq-hosted Llama 3.1 for explanation generation
- Provides human-readable reasoning behind detected anomalies

### Telegram Alerts
- Real-time notifications
- AI-generated explanations included
- Supports multiple concurrent users

### Monitoring Dashboard
Built using Streamlit:

- Historical anomaly tracking
- Event filtering
- Timeline visualization
- Alert review interface

### Event Logging
- JSONL-based event storage
- Historical anomaly persistence
- Easy integration with analytics pipelines

---

# 🏗 System Architecture

```text
                    Yahoo Finance API
                            |
                            V
                    Price Collection
                            |
                            V
                  Ensemble Detection
                            |
        ------------------------------------------------
        |        |         |        |         |        |
        V        V         V        V         V        V
     ZScore  IForest    LOF     OCSVM   Residual   LSTM AE
                            |
                     Voting Mechanism
                            |
                     Anomaly Detected?
                            |
                           YES
                            |
                            V
                     Serper News Search
                            |
                            V
                      Groq Llama 3.1
                            |
                            V
                   Natural Language Explanation
                            |
                            V
                     Telegram Notification
                            |
                            V
                        Event Log
                            |
                            V
                    Streamlit Dashboard
```

---

# 🧠 Detection Pipeline

## 1. Market Data Collection

The system retrieves live stock data using:

- Yahoo Finance API (`yfinance`)
- 1-minute interval price updates
- Dynamic ticker validation

---

## 2. Ensemble Anomaly Detection

Multiple independent detectors analyze the incoming price stream.

### Z-Score Detector

Detects statistical outliers:

```
z = (x - μ) / σ
```

---

### Isolation Forest

Tree-based anomaly detection that isolates unusual observations.

---

### Local Outlier Factor (LOF)

Measures how different a point is relative to neighboring points.

---

### One-Class SVM

Learns normal behavior and identifies deviations.

---

### Residual Error Detector

Compares current price against rolling averages.

---

### LSTM Autoencoder

Sequence-based deep learning model that:

- Learns normal market patterns
- Reconstructs price sequences
- Flags high reconstruction errors as anomalies

---

## 3. Voting Mechanism

An anomaly is generated when at least 3 detectors agree.

```python
if anomaly_votes >= 3:
    trigger_alert()
```

This ensemble approach improves robustness and reduces false positives.

---

# 🤖 AI Explanation Layer

After anomaly detection:

### News Retrieval

Relevant headlines are retrieved using:

- Serper Search API

Example:

```text
AAPL stock earnings news
```

---

### LLM Explanation

The retrieved context is sent to:

- Llama 3.1 8B Instant
- Hosted on Groq

The model generates:

- Possible causes
- Market interpretation
- Human-readable explanations

---

# 📬 Telegram Alerting

Example Alert:

```text
🚨 Stock Anomaly Detected

Symbol: AAPL
Price: $210.42

Possible Cause:
Apple announced stronger-than-expected earnings,
leading to unusual market activity.

AI Explanation:
The anomaly appears correlated with increased
investor buying following positive earnings guidance.
```

---

# 📊 Dashboard

The Streamlit dashboard provides:

- Historical anomaly records
- Symbol-based filtering
- Timeline analysis
- Explanation review

---

# 📁 Project Structure

```text
Automated-Trade-Monitoring-Agent/
│
├── app.py
├── dashboard.py
├── monitor.py
├── detector.py
├── lstm_autoencoder.py
├── news_service.py
├── telegram_service.py
├── utils/
│
├── models/
│
├── data/
│
├── events_log.jsonl
│
├── requirements.txt
│
└── README.md
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/Ayushsharma832/Automated-Trade-Monitoring-Agent.git

cd Automated-Trade-Monitoring-Agent
```

---

## Create Virtual Environment

```bash
python -m venv venv
```

Activate:

### Windows

```bash
venv\Scripts\activate
```

### Linux / Mac

```bash
source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key

SERPER_API_KEY=your_serper_key

TELEGRAM_BOT_TOKEN=your_bot_token
```

---

# ▶️ Running the Application

## Start Monitoring Service

```bash
python app.py
```

---

## Launch Dashboard

```bash
streamlit run dashboard.py
```

---

# ☁️ Deployment

The application can be deployed on:

- AWS EC2
- Docker
- Azure VM
- GCP Compute Engine

Current deployment target:

```text
AWS EC2
```

---

# 📈 Future Improvements

### Retrieval & Explanations

- News reranking
- Multi-source news aggregation
- Confidence scoring

### Detection

- Adaptive anomaly thresholds
- Online learning
- Dynamic model retraining
- Feature engineering (Volume, RSI, MACD)

### Scalability

- Redis-backed state management
- Kafka event streaming
- Celery task queues
- Async processing

### Observability

- Prometheus metrics
- Grafana dashboards
- ML model monitoring

---

# ⚠️ Limitations

- Uses Yahoo Finance (not institutional-grade market data)
- LSTM model requires periodic retraining
- Thread-per-user monitoring limits scalability
- Limited feature engineering
- Equal-weight ensemble voting

---

# 🛠 Tech Stack

### Backend

- Python

### Data Collection

- yfinance

### Machine Learning

- Scikit-learn
- TensorFlow / Keras

### Deep Learning

- LSTM Autoencoder

### LLM

- Groq
- Llama 3.1

### Search

- Serper API

### Notifications

- Telegram Bot API

### Dashboard

- Streamlit

### Deployment

- AWS EC2

---

# 🎯 Resume Highlights

- Built an AI-powered trade surveillance platform using ensemble anomaly detection and LSTM Autoencoders.
- Integrated market news retrieval and LLM-generated explanations using Groq-hosted Llama 3.1.
- Implemented real-time Telegram alerting and monitoring workflows.
- Developed a Streamlit dashboard for anomaly visualization and historical analysis.
- Deployed the solution on AWS EC2 for continuous monitoring.

---

## Author

Ayush Sharma

GitHub: https://github.com/Ayushsharma832

LinkedIn: https://linkedin.com/in/your-profile
