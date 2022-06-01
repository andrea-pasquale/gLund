import optuna
from simpler_gaussian import build_and_train_model

def objective(trial):
    lr_d = trial.suggest_loguniform("discriminator learning rate", 0.001, 0.1)
    lr = trial.suggest_loguniform("generator learning rate", 0.001, 0.1)
    batch_samples = trial.suggest_categorical("batch_samples", [32, 64, 128])
    n_epochs = trial.suggest_categorical("number of epochs", [2000, 4000])
    return build_and_train_model(lr_d=lr_d, lr=lr, batch_samples=batch_samples, n_epochs=n_epochs)

if __name__ == '__main__':
    study = optuna.load_study(study_name='hyperopt2',
                              storage='mysql://root:9FsbrzY5FQ@galileo/example')
    study.optimize(objective, n_trials=2)
