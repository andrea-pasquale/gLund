import optuna
import json
from simpler_gaussian import build_and_train_model
import argparse

def objective(trial):
    lr_d = trial.suggest_loguniform("discriminator learning rate", 0.001, 0.1)
    lr = trial.suggest_loguniform("generator learning rate", 0.001, 0.1)
    batch_samples = trial.suggest_categorical("batch_samples", [32, 64, 128])
    n_epochs = trial.suggest_categorical("number of epochs", [2000, 10000, 20000])
    return build_and_train_model(lr_d=lr_d, lr=lr, batch_samples=batch_samples, n_epochs=n_epochs)

study = optuna.create_study(study_name='qgan 8 digits images', direction='minimize')
study.optimize(objective, n_trials=2, n_jobs=-1)
best_params = study.best_params
with open('best_params.json', 'w') as fo:
    json.dump(best_params, fo)