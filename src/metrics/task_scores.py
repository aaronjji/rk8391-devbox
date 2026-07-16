"""Official Task 1/2/3 composite score formulas.

Task1 == Task2 formula (per the challenge spec):
  Score = 10 x [ 0.3 x (0.3*Sen + 0.3*Spec + 0.4*Acc)
                 + 0.4 x DSC
                 + 0.3 x (0.5*COR + 0.5*(1-INF)) ]

Task3: per-biomarker Score_i = 0.5*P(MAE_i) + 0.5*Q(SMAPE_i), P/Q unknown
organizer-defined negative-linear normalizations to [0,1] -- NOT implemented
here (see task3_score_raw for the MAE/SMAPE-only proxy to optimize against
until the real scale is backed out from real leaderboard submissions).
"""
import numpy as np


def task1_task2_score(dsc: float, sen: float, spec: float, acc: float, cor: float, inf: float) -> float:
    classification = 0.3 * sen + 0.3 * spec + 0.4 * acc
    topology = 0.5 * cor + 0.5 * (1 - inf)
    return 10 * (0.3 * classification + 0.4 * dsc + 0.3 * topology)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    denom = np.where(denom == 0, 1e-8, denom)
    return float(100 * np.mean(np.abs(y_true - y_pred) / denom))


def task3_score_raw(biomarker_mae: dict, biomarker_smape: dict) -> dict:
    """Raw MAE/SMAPE per biomarker -- the real optimization target until
    P()/Q() are backed out from real submissions (see plan doc, Section 2)."""
    return {"mae": biomarker_mae, "smape": biomarker_smape}


def round_score(score_task1: float, score_task2: float, score_task3: float) -> float:
    return 0.2 * score_task1 + 0.4 * score_task2 + 0.4 * score_task3


def total_score(score_preliminary: float, score_final: float) -> float:
    return 0.3 * score_preliminary + 0.7 * score_final
