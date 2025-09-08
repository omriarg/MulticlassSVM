import pandas as pd
import torch

class data_init:
    def __init__(self, data, train_ratio=0.70, cv_ratio=0.15):

        data = data.copy()

        # --- 2) Feature engineering + labels ---
        features = self.extract_features(data)

        if 'Close' in data.columns:
            data['PriceChange'] = data['Close'].pct_change() * 100
            # learn quantile-based thresholds (tertiles) on available PriceChange
            pc = data['PriceChange'].dropna()
            if len(pc) >= 3:
                q_low = float(pc.quantile(1/3))
                q_high = float(pc.quantile(2/3))
            else:
                # fallback if not enough data to compute quantiles
                q_low, q_high = -0.5, 0.5

            self.thresholds = {'low': q_low, 'high': q_high, 'quantiles': (1/3, 2/3)}
            print(f"Quantile thresholds (PriceChange %): low={q_low:.4f}, high={q_high:.4f}")

            features['Label'] = data['PriceChange'].apply(self.classify_multiclass)  # uses learned thresholds

        elif 'Label' in data.columns:
            features['Label'] = data['Label']
            self.thresholds = {'low': -0.5, 'high': 0.5, 'quantiles': None}
        else:
            raise ValueError("Label not found and cannot compute from Close price")
        features = features.dropna().astype(float)

        self.features_df = features.drop(columns=['Label'])
        self.labels_sr = features['Label'].astype(int)

        print("\nClass distribution:")
        print(self.labels_sr.value_counts().sort_index())
        print("-" * 40)

        # --- 3) Chronological split ---
        self.train_ratio = train_ratio
        self.cv_ratio = cv_ratio
        (self.X_train, self.Y_train,
         self.X_cv, self.Y_cv,
         self.X_test, self.Y_test) = self.train_cross_test_split()

        # --- 4) Fit scaler on TRAIN only, transform all splits ---
        self.feature_mean, self.feature_std = self._fit_train_scaler(self.X_train)
        self.X_train = self._transform_with_train_stats(self.X_train)
        self.X_cv    = self._transform_with_train_stats(self.X_cv)
        self.X_test  = self._transform_with_train_stats(self.X_test)

        # Convenience tensors (raw, not scaled)
        self.features_tensor = torch.tensor(self.features_df.values, dtype=torch.float32)
        self.labels_tensor   = torch.tensor(self.labels_sr.values, dtype=torch.long)

    # ----------prepare external dataframe (unseen data, after proccessing initial (training) data ----------
    def prepare_new_df(self, data):
        if not hasattr(self, "feature_mean") or not hasattr(self, "feature_std"):
            raise ValueError("Scaler not fitted yet. Initialize on training data first.")

        df = data.copy()
        rename_map = {
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'close_price': 'Close',
            'volume': 'Volume',
        }
        df.rename(columns=rename_map, inplace=True)

        drop_cols = [
            'date', 'RSI_14', 'SMA_20', 'OBV', 'ATRr_14',
            'rolling_mean_close_5', 'target_close'
        ]
        df = df.drop(columns=drop_cols, errors='ignore')

        feats = self.extract_features(df)
        if 'Close' in df.columns:
            df['PriceChange'] = df['Close'].pct_change() * 100
            feats['Label'] = df['PriceChange'].apply(self.classify_multiclass)
        elif 'Label' in df.columns:
            feats['Label'] = df['Label']
        else:
            raise ValueError("Label not found and cannot compute from Close price")

        feats = feats.dropna().astype(float)

        # align schema
        X_df = feats.drop(columns=['Label'])
        missing = set(self.features_df.columns) - set(X_df.columns)
        if missing:
            raise ValueError(f"New data missing feature columns: {sorted(missing)}")
        X_df = X_df[self.features_df.columns]
        Y_sr = feats['Label'].astype(int)

        # tensors + scale
        X = torch.tensor(X_df.values, dtype=torch.float32)
        Y = torch.tensor(Y_sr.values, dtype=torch.long)
        X = self._transform_with_train_stats(X)
        return X, Y

    # ---------- Split ----------
    def train_cross_test_split(self):
        X = torch.tensor(self.features_df.values, dtype=torch.float32)
        Y = torch.tensor(self.labels_sr.values, dtype=torch.long)

        n = X.shape[0]
        train_end = int(self.train_ratio * n)
        cv_end    = int((self.train_ratio + self.cv_ratio) * n)

        return X[:train_end], Y[:train_end], X[train_end:cv_end], Y[train_end:cv_end], X[cv_end:], Y[cv_end:]
    #Feature scaling
    @staticmethod
    def _fit_train_scaler(X_train):
        mean = X_train.mean(dim=0, keepdim=True)
        std  = X_train.std(dim=0, keepdim=True).clamp_min(1e-8)
        return mean, std

    def _transform_with_train_stats(self, X):
        return (X - self.feature_mean) / self.feature_std

    # ---------- Feature engineering ----------
    @staticmethod
    def extract_features(data):
        df = pd.DataFrame()
        df['close_threedays_ago'] = data['Close'].shift(3)
        df['close_yesterday']     = data['Close'].shift(1)
        df['close_last_week']     = data['Close'].shift(5)
        df['close_last_month']    = data['Close'].shift(21)

        sma_5  = data['Close'].shift(1).rolling(window=5).mean()
        sma_20 = data['Close'].shift(1).rolling(window=20).mean()
        df['sma5_sma20_ratio'] = sma_5 / sma_20

        high_low   = data['High'].shift(1) - data['Low'].shift(1)
        high_close = (data['High'].shift(1) - data['Close'].shift(2)).abs()
        low_close  = (data['Low'].shift(1)  - data['Close'].shift(2)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(window=14).mean()

        df['return_1d']  = data['Close'].shift(1) / data['Close'].shift(2) - 1
        df['return_5d']  = data['Close'].shift(1) / data['Close'].shift(6) - 1
        df['return_21d'] = data['Close'].shift(1) / data['Close'].shift(22) - 1

        past_returns = data['Close'].shift(1).pct_change()
        df['volatility_5']  = past_returns.rolling(window=5).std()
        df['volatility_21'] = past_returns.rolling(window=21).std()

        df['volume_mean_5']   = data['Volume'].shift(1).rolling(window=5).mean()
        df['volume_mean_20']  = data['Volume'].shift(1).rolling(window=20).mean()
        df['volume_ratio_5_20'] = df['volume_mean_5'] / df['volume_mean_20']

        return df.dropna()

    def classify_multiclass(self,change):
        # use learned quantile thresholds
        low = (self.thresholds['low'] if hasattr(self, 'thresholds') and 'low' in self.thresholds else -0.5)
        high = (self.thresholds['high'] if hasattr(self, 'thresholds') and 'high' in self.thresholds else 0.5)

        if pd.isna(change):
            return None  #
        if change > high:
            return 2
        elif change > low:
            return 1
        else:
            return 0
    # ---------- Accessors ----------
    def get_features_tensor(self):
        return self.features_tensor

    def get_labels_tensor(self):
        return self.labels_tensor

    def get_splits(self):
        return (self.X_train, self.Y_train), (self.X_cv, self.Y_cv), (self.X_test, self.Y_test)
