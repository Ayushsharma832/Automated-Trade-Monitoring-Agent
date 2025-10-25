# scorer.py
import numpy as np
import pandas as pd
from collections import deque
from scipy.stats import zscore
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import datetime
import warnings
warnings.filterwarnings("ignore")


class EnsembleScorer:
    def __init__(self, window_size=60):
        """
        Ensemble anomaly detector using statistical, ML, and DL models.
        """
        self.window_size = window_size
        self.price_history = {}  # {symbol: deque([...])}

        # Initialize models (LSTM will be built per symbol later)
        self.models = {
            "iforest": IsolationForest(contamination=0.05, random_state=42),
            "lof": LocalOutlierFactor(n_neighbors=20, contamination=0.05),
            "ocsvm": OneClassSVM(nu=0.05, kernel="rbf", gamma="auto")
        }

        self.lstm_models = {}  # {symbol: trained LSTM autoencoder}

    # ----------------- Helper: Build LSTM Autoencoder -----------------
    def _build_lstm_autoencoder(self, input_shape):
        model = Sequential([
            LSTM(32, activation='relu', input_shape=input_shape, return_sequences=False),
            Dropout(0.2),
            Dense(input_shape[0], activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model

    # ----------------- Helper: Train LSTM model -----------------
    def _train_lstm(self, symbol, data):
        data = np.array(data)
        seq_len = min(10, len(data))  # sequence length for LSTM
        X = np.array([data[i-seq_len:i] for i in range(seq_len, len(data))])
        model = self._build_lstm_autoencoder((seq_len, 1))
        model.fit(X.reshape(X.shape[0], X.shape[1], 1), X.reshape(X.shape[0], X.shape[1]),
                  epochs=15, batch_size=4, verbose=0)
        self.lstm_models[symbol] = (model, seq_len)

    # ----------------- Main scoring logic -----------------
    def update_and_score(self, symbol, price, timestamp=None):
        if timestamp is None:
            timestamp = datetime.datetime.utcnow()

        # maintain price history
        if symbol not in self.price_history:
            self.price_history[symbol] = deque(maxlen=self.window_size)
        self.price_history[symbol].append(price)

        if len(self.price_history[symbol]) < 20:
            return {
                "symbol": symbol,
                "timestamp": timestamp.isoformat(),
                "price": price,
                "ready": False,
                "message": "Not enough data to score yet."
            }

        data = np.array(self.price_history[symbol]).reshape(-1, 1)

        # ---- 1. Z-SCORE ----
        z_scores = np.abs(zscore(data))
        z_flag = z_scores[-1] > 3

        # ---- 2. ISOLATION FOREST ----
        self.models["iforest"].fit(data)
        iforest_flag = self.models["iforest"].predict(data)[-1] == -1

        # ---- 3. RESIDUAL ERROR ----
        rolling_mean = pd.Series(data.flatten()).rolling(window=10).mean().iloc[-1]
        residual = abs(price - rolling_mean)
        residual_flag = residual > (2 * np.std(data))  # 2σ deviation

        # ---- 4. LOCAL OUTLIER FACTOR ----
        lof_preds = self.models["lof"].fit_predict(data)
        lof_flag = lof_preds[-1] == -1

        # ---- 5. ONE-CLASS SVM ----
        self.models["ocsvm"].fit(data)
        ocsvm_flag = self.models["ocsvm"].predict(data)[-1] == -1

        # ---- 6. LSTM AUTOENCODER ----
        if symbol not in self.lstm_models:
            self._train_lstm(symbol, data)
        model, seq_len = self.lstm_models[symbol]
        if len(data) > seq_len:
            X_test = np.array([data[-seq_len:]]).reshape(1, seq_len, 1)
            reconstruction = model.predict(X_test, verbose=0)
            loss = np.mean((X_test - reconstruction) ** 2)
            lstm_flag = loss > np.percentile(loss, 95)  # top 5% = anomaly
        else:
            lstm_flag = False

        # ---- Combine results ----
        model_flags = sum([z_flag, iforest_flag, residual_flag, lof_flag, ocsvm_flag, lstm_flag])
        final_anomaly = model_flags >= 3  # at least half

        return {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "price": float(price),
            "scores": {
                "zscore_flag": bool(z_flag),
                "iforest_flag": bool(iforest_flag),
                "residual_flag": bool(residual_flag),
                "lof_flag": bool(lof_flag),
                "ocsvm_flag": bool(ocsvm_flag),
                "lstm_flag": bool(lstm_flag)
            },
            "model_flags": int(model_flags),
            "final_anomaly": bool(final_anomaly),
            "ready": True
        }
