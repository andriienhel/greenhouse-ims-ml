"""
Моделі прогнозування з єдиним інтерфейсом fit/predict.

Набір дібрано так, щоб охопити принципово різні підходи і показати, що вибір
на користь складної моделі обґрунтований, а не зроблений за інерцією:

  Наївний       прогноз «величина не зміниться». Обов'язкова точка відліку:
                на горизонті 15 хвилин інерція теплиці велика, і будь-яка
                модель має довести, що вона краща за це.
  Ridge         лінійна регресія з регуляризацією. Показує, скільки можна
                видобути суто лінійною залежністю.
  ВипадковийЛіс ансамбль дерев, уловлює нелінійності без налаштування.
  ГрадієнтнийБустинг  зазвичай найсильніша модель на табличних даних такого
                обсягу.
  GRU           рекурентна нейромережа, працює з послідовністю цілком.

Усі моделі, крім наївної, навчаються на спостережуваних (спотворених) даних —
так само, як це робив би реальний прилад.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class BaseModel:
    """Спільний інтерфейс усіх моделей."""

    name = "base"

    def fit(self, X, y, X_val=None, y_val=None):  # noqa: D102
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:  # noqa: D102
        raise NotImplementedError


class Persistence(BaseModel):
    """Наївний прогноз: значення через h відліків дорівнює поточному.

    Модель нічого не вчить. Вона бере ознаку з нульовим лагом цільового каналу
    і видає її як прогноз.
    """

    name = "Наївний"

    def __init__(self, target: str):
        self.col = f"{target}_lag0"

    def fit(self, X, y, X_val=None, y_val=None):
        if self.col not in X.columns:
            raise KeyError(f"В ознаках немає стовпця {self.col}")
        return self

    def predict(self, X) -> np.ndarray:
        return X[self.col].to_numpy(dtype=float)


class SklearnModel(BaseModel):
    """Обгортка над моделями scikit-learn із нормуванням ознак."""

    def __init__(self, estimator, name: str, scale: bool = False):
        self.estimator = estimator
        self.name = name
        self.scale = scale
        self._scaler = None

    def fit(self, X, y, X_val=None, y_val=None):
        Xv = np.asarray(X, dtype=float)
        if self.scale:
            from sklearn.preprocessing import StandardScaler

            self._scaler = StandardScaler().fit(Xv)
            Xv = self._scaler.transform(Xv)
        self.estimator.fit(Xv, np.asarray(y, dtype=float))
        return self

    def predict(self, X) -> np.ndarray:
        Xv = np.asarray(X, dtype=float)
        if self._scaler is not None:
            Xv = self._scaler.transform(Xv)
        return self.estimator.predict(Xv)


def make_ridge(alpha: float = 1.0, seed: int = 42) -> SklearnModel:
    from sklearn.linear_model import Ridge

    return SklearnModel(Ridge(alpha=alpha), "Ridge", scale=True)


def make_random_forest(seed: int = 42, n_jobs: int = 1) -> SklearnModel:
    """Випадковий ліс.

    УВАГА, n_jobs=1 обрано навмисно. На macOS розпаралелювання joblib усередині
    RandomForest конфліктує з уже завантаженою бібліотекою OpenMP (її тягне за
    собою xgboost): процес іде у вічне очікування і не завершується. Навчання в
    один потік трохи повільніше, зате не зависає. Якщо знадобиться прискорення,
    збільшувати n_jobs лише разом із перевіркою на конкретній машині.
    """
    from sklearn.ensemble import RandomForestRegressor

    est = RandomForestRegressor(
        n_estimators=150,
        max_depth=14,
        min_samples_leaf=4,
        # За замовчуванням у задачі регресії кожен вузол перебирає всі ознаки,
        # а їх тут понад шістдесят — навчання виходить дуже довгим. Половина
        # ознак на вузол дає кратне прискорення і додатково робить дерева менш
        # схожими одне на одне, що ансамблю тільки на користь.
        max_features=0.5,
        random_state=seed,
        n_jobs=n_jobs,
    )
    return SklearnModel(est, "ВипадковийЛіс")


def make_hist_gb(seed: int = 42) -> SklearnModel:
    """Градієнтний бустинг на гістограмах (реалізація scikit-learn).

    Той самий алгоритм, що в LightGBM: ознаки попередньо розбиваються на 256
    кошиків, тому навчання йде в рази швидше за класичний бустинг майже без
    втрати точності.

    ЧОМУ НЕ XGBOOST. На macOS xgboost і torch підтягують різні реалізації
    OpenMP, і за спільної роботи в одному процесі настає взаємне блокування:
    процес живий, але не виконує жодної інструкції. Перевірено — зависає і
    xgboost після імпорту torch, і навчання мережі після xgboost. Реалізація зі
    scikit-learn цієї проблеми позбавлена, а точність відрізняється незначно (на
    контрольній вибірці RMSE 0,451 проти 0,427 у xgboost для температури на
    горизонті 15 хвилин).
    """
    from sklearn.ensemble import HistGradientBoostingRegressor

    est = HistGradientBoostingRegressor(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )
    return SklearnModel(est, "ГрадієнтнийБустинг")


def make_xgboost(seed: int = 42, n_jobs: int = 4) -> SklearnModel:
    """XGBoost. УВАГА: несумісний із torch в одному процесі на macOS.

    Залишено для звіряння результатів, в основних дослідах не використовується.
    Запускати лише в процесі, де torch не імпортується.
    """
    from xgboost import XGBRegressor

    est = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=n_jobs,
        tree_method="hist",
    )
    return SklearnModel(est, "XGBoost")


class GRUModel(BaseModel):
    """Рекурентна нейромережа на GRU-комірках.

    На відміну від табличних моделей, отримує не набір окремих лагів, а всю
    послідовність спостережень і сама вирішує, що з історії важливе.

    Матриця ознак подається як є: мережа бачить один вектор ознак на відлік, а
    пам'ять про минуле зберігає у внутрішньому стані. Така схема простіша в
    супроводі, ніж окремий конвеєр послідовностей, і дозволяє порівнювати
    моделі на однакових ознаках — інакше порівняння було б некоректним.
    """

    name = "GRU"

    def __init__(
        self,
        hidden: int = 64,
        seq_len: int = 24,
        epochs: int = 40,
        batch_size: int = 128,
        lr: float = 1e-3,
        patience: int = 6,
        seed: int = 42,
    ):
        self.hidden = hidden
        self.seq_len = seq_len
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self._net = None
        self._scaler = None
        self._y_mean = 0.0
        self._y_std = 1.0

    # ------------------------------------------------------------------
    def _to_sequences(self, X: np.ndarray) -> np.ndarray:
        """Нарізати матрицю ознак на перекривні вікна довжини seq_len.

        Перші seq_len-1 відліків доповнюються повтором першого рядка, щоб
        кількість вікон збігалася з кількістю відліків.
        """
        n, f = X.shape
        pad = np.repeat(X[:1], self.seq_len - 1, axis=0)
        padded = np.vstack([pad, X])
        idx = np.arange(self.seq_len)[None, :] + np.arange(n)[:, None]
        return padded[idx]

    def _build(self, n_features: int):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(self, n_in: int, hidden: int):
                super().__init__()
                self.gru = nn.GRU(n_in, hidden, batch_first=True)
                self.head = nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.ReLU(),
                    nn.Linear(hidden // 2, 1),
                )

            def forward(self, x):
                out, _ = self.gru(x)
                return self.head(out[:, -1, :]).squeeze(-1)

        return Net(n_features, self.hidden)

    def fit(self, X, y, X_val=None, y_val=None):
        import torch
        from sklearn.preprocessing import StandardScaler

        # Один потік: мережа тут невелика, виграшу від багатопотоковості немає,
        # а ризик конфлікту з іншими бібліотеками, що використовують OpenMP, є
        torch.set_num_threads(1)

        Xv = np.asarray(X, dtype=float)
        yv = np.asarray(y, dtype=float)

        self._scaler = StandardScaler().fit(Xv)
        # Цільову змінну теж нормуємо: інакше градієнти для величин різного
        # масштабу (°C і ppm) поводяться зовсім по-різному
        self._y_mean = float(np.mean(yv))
        self._y_std = float(np.std(yv)) or 1.0

        Xs = self._to_sequences(self._scaler.transform(Xv))
        ys = (yv - self._y_mean) / self._y_std

        self._net = self._build(Xv.shape[1])
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        Xt = torch.tensor(Xs, dtype=torch.float32)
        yt = torch.tensor(ys, dtype=torch.float32)

        has_val = X_val is not None and y_val is not None
        if has_val:
            Xvl = self._to_sequences(
                self._scaler.transform(np.asarray(X_val, dtype=float)))
            yvl = (np.asarray(y_val, dtype=float) - self._y_mean) / self._y_std
            Xvt = torch.tensor(Xvl, dtype=torch.float32)
            yvt = torch.tensor(yvl, dtype=torch.float32)

        best = float("inf")
        best_state = None
        bad_epochs = 0
        n = Xt.shape[0]

        for _ in range(self.epochs):
            self._net.train()
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                sel = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self._net(Xt[sel]), yt[sel])
                loss.backward()
                opt.step()

            if not has_val:
                continue

            # Рання зупинка за перевірочною вибіркою: без неї мережа
            # перенавчається на 17 тисячах відліків дуже швидко
            self._net.eval()
            with torch.no_grad():
                v = float(loss_fn(self._net(Xvt), yvt))
            if v < best - 1e-5:
                best = v
                best_state = {k: t.clone() for k, t in self._net.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, X) -> np.ndarray:
        import torch

        Xs = self._to_sequences(self._scaler.transform(np.asarray(X, dtype=float)))
        self._net.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, Xs.shape[0], 512):
                batch = torch.tensor(Xs[i : i + 512], dtype=torch.float32)
                preds.append(self._net(batch).numpy())
        return np.concatenate(preds) * self._y_std + self._y_mean


def default_models(target: str, seed: int = 42, with_gru: bool = True) -> list[BaseModel]:
    """Стандартний набір моделей для порівняння в дослідах."""
    models: list[BaseModel] = [
        Persistence(target),
        make_ridge(seed=seed),
        make_random_forest(seed=seed),
        make_hist_gb(seed=seed),
    ]
    if with_gru:
        models.append(GRUModel(seed=seed))
    return models
