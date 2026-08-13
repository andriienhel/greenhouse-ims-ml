"""
Метрики точності прогнозу та статистична перевірка результатів.

Окремий модуль потрібен тому, що недостатньо показати «у нас RMSE менша».
Рецензент має право спитати, чи значуща різниця між моделями і чи не отримана
вона випадково на конкретній вибірці. Тому тут є і метрики, і перевірка
статистичної значущості різниці прогнозів.
"""
from __future__ import annotations

import numpy as np


def _clean(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(t) & np.isfinite(p)
    return t[ok], p[ok]


def rmse(y_true, y_pred) -> float:
    """Середньоквадратична похибка прогнозу."""
    t, p = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def mae(y_true, y_pred) -> float:
    """Середня абсолютна похибка."""
    t, p = _clean(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def mape(y_true, y_pred, eps: float = 1e-9) -> float:
    """Середня абсолютна похибка у відсотках.

    Обережно: метрика нестійка, коли величина проходить через нуль (наприклад,
    освітленість уночі). Для таких каналів спиратися слід на RMSE.
    """
    t, p = _clean(y_true, y_pred)
    denom = np.maximum(np.abs(t), eps)
    return float(np.mean(np.abs((p - t) / denom)) * 100)


def bias(y_true, y_pred) -> float:
    """Систематична складова похибки прогнозу."""
    t, p = _clean(y_true, y_pred)
    return float(np.mean(p - t))


def r2(y_true, y_pred) -> float:
    """Коефіцієнт детермінації."""
    t, p = _clean(y_true, y_pred)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def skill_score(y_true, y_pred, y_baseline) -> float:
    """Відносне покращення щодо еталонного прогнозу, %.

    Показує, наскільки модель краща за наївний прогноз «значення не
    зміниться». Це чесніше за абсолютну RMSE: на короткому горизонті інерція
    об'єкта настільки велика, що наївний прогноз уже дуже добрий, і модель
    зобов'язана довести, що вона взагалі потрібна.
    """
    m = rmse(y_true, y_pred)
    b = rmse(y_true, y_baseline)
    return float((1 - m / b) * 100) if b > 0 else float("nan")


def evaluate(y_true, y_pred, y_baseline=None) -> dict[str, float]:
    """Повний набір метрик одним викликом."""
    out = {
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred),
        "зміщення": bias(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    if y_baseline is not None:
        out["виграш над наївним,%"] = skill_score(y_true, y_pred, y_baseline)
    return out


def diebold_mariano(
    y_true, y_pred_a, y_pred_b, horizon: int = 1
) -> tuple[float, float]:
    """Тест Діболда — Маріано на рівність точності двох прогнозів.

    Перевіряє нульову гіпотезу «обидві моделі прогнозують однаково точно».
    Повертає (статистика, p-значення). Від'ємна статистика означає, що модель A
    точніша за модель B.

    Звичайний t-тест тут незастосовний: похибки прогнозу сусідніх моментів часу
    корельовані, і він завищив би значущість. Тест Діболда — Маріано враховує
    цю автокореляцію через оцінку довготривалої дисперсії (Ньюї — Веста).
    """
    from scipy import stats

    t = np.asarray(y_true, dtype=float)
    a = np.asarray(y_pred_a, dtype=float)
    b = np.asarray(y_pred_b, dtype=float)
    ok = np.isfinite(t) & np.isfinite(a) & np.isfinite(b)
    t, a, b = t[ok], a[ok], b[ok]

    # Ряд різниць квадратів похибок двох моделей
    d = (a - t) ** 2 - (b - t) ** 2
    n = d.size
    d_mean = float(np.mean(d))

    # Довготривала дисперсія з поправкою на автокореляцію до лагу h-1
    gamma0 = float(np.mean((d - d_mean) ** 2))
    var = gamma0
    for lag in range(1, max(1, horizon)):
        if lag >= n:
            break
        cov = float(np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean)))
        var += 2 * cov
    var = max(var, 1e-30)

    dm = d_mean / np.sqrt(var / n)
    # Поправка Харві — Лейборна — Ньюболда на малу вибірку
    if n > horizon:
        corr = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
        dm *= corr
    p_value = float(2 * (1 - stats.t.cdf(abs(dm), df=max(1, n - 1))))
    return float(dm), p_value
