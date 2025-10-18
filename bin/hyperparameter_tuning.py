#!/usr/bin/env python3

# stdlib imports
import argparse

# pip imports
import optuna

# local imports
from bin.train import TrainCommand
from src.utils.model_utils import ModelUtils


class HyperParameterTuning:
    def __init__(self, max_file_count: int = 20, max_epoch_count: int = 5, verbose: bool = True, trial_count: int = 10) -> None:
        self._max_file_count = max_file_count
        self._max_epoch_count = max_epoch_count
        self._verbose = verbose
        self._trial_count = trial_count

    def run(self) -> None:
        """Perform hyperparameter tuning using Optuna."""
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self._trial_count)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for Optuna to minimize."""

        # define hyperparameters to tune
        learning_rate: float | None = None
        batch_size: int | None = None
        train_test_split_ratio: float | None = None
        early_stopping_patience: int | None = None
        early_stopping_threshold: float | None = None
        scheduler_patience: int | None = None
        scheduler_threshold: float | None = None

        # set values which wont be tuned
        # learning_rate = 0.00025
        batch_size = 32
        train_test_split_ratio = 0.7

        # define the hyperparameters to be tuned by optuna
        if learning_rate is None:
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        if batch_size is None:
            batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        if train_test_split_ratio is None:
            train_test_split_ratio = trial.suggest_categorical("train_test_split_ratio", [0.6, 0.7, 0.8, 0.9])
        if early_stopping_patience is None:
            early_stopping_patience = trial.suggest_int("early_stopping_patience", 5, 20)
        if early_stopping_threshold is None:
            early_stopping_threshold = trial.suggest_float("early_stopping_threshold", 0.001, 0.1, log=True)
        if scheduler_patience is None:
            scheduler_patience = trial.suggest_int("scheduler_patience", 2, 10)
        if scheduler_threshold is None:
            scheduler_threshold = trial.suggest_float("scheduler_threshold", 0.001, 0.1, log=True)

        # Train the model with the suggested hyperparameters
        model_accuracy_final = TrainCommand.train(
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_test_split_ratio=train_test_split_ratio,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            scheduler_patience=scheduler_patience,
            scheduler_threshold=scheduler_threshold,
            max_file_count=self._max_file_count,
            # fixed parameters
            model_name=ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D,
            max_epoch_count=self._max_epoch_count,
            verbose=self._verbose,
            random_seed=100,
        )

        # Return the final model accuracy as the objective value to maximize
        return model_accuracy_final


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    argParser = argparse.ArgumentParser(description="Perform hyperparameter tuning using Optuna.")

    hyper_parameter_tuning = HyperParameterTuning(verbose=False, trial_count=20, max_file_count=15, max_epoch_count=10)
    hyper_parameter_tuning.run()
