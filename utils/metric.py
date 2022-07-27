import torch
from functools import partial
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, r2_score, mean_absolute_error, mean_squared_error
from imblearn.metrics import macro_averaged_mean_absolute_error


def poissonLoss(xbeta, y):
    """Custom loss function for Poisson model."""
    loss=torch.mean(torch.exp(xbeta)-y*xbeta)
    return loss

reg_loss_fns = {
    'mse': torch.nn.MSELoss(),
    'mae': torch.nn.L1Loss(),
    'huber': torch.nn.HuberLoss(),
    'poisson': torch.nn.PoissonNLLLoss()
}

def macro_mae(y_true, y_pred):
    # Round predictions to the nearest integer
    return macro_averaged_mean_absolute_error(y_true, torch.round(y_pred))

criteria_fns = {
    'MAE': mean_absolute_error,
    'MSE': mean_squared_error,
    'RMSE': partial(mean_squared_error, squared=False),
    'Macro MAE': macro_mae,
    'R2': r2_score,
    'Accuracy': accuracy_score,
    'Macro Recall': balanced_accuracy_score,
    'F1': f1_score,
    'Micro F1': partial(f1_score, average='micro'),
    'Weighted F1': partial(f1_score, average='weighted'),
    'MCC': matthews_corrcoef
}