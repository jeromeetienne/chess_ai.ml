#!/usr/bin/env python3

# stdlib imports
import argparse
import dataclasses
import os
import datetime
from typing import Sequence

# pip imports
import optuna
import optunahub

# local imports
from bin.train import TrainCommand
from src.libs.chess_model import ChessModelParams, ChessModelConv2dParams, ChessModelResnetParams
from src.utils.model_utils import ModelUtils

__dirname__ = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.abspath(os.path.join(__dirname__, "..", "output"))
model_path = os.path.abspath(os.path.join(output_path, "model"))


class HyperParameterTuning:
    def __init__(self, model_name: str, max_file_count: int = 20, max_epoch_count: int = 5, verbose: bool = True, trial_count: int = 10) -> None:
        self._model_name = model_name
        self._max_file_count = max_file_count
        self._max_epoch_count = max_epoch_count
        self._verbose = verbose
        self._trial_count = trial_count

    def run(self) -> None:
        """Perform hyperparameter tuning using Optuna."""

        # =============================================================================
        # Study
        # =============================================================================

        # get time in the form YYYYMMDD_HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{timestamp}_chess_ai.ml_fc{self._max_file_count}_me{self._max_epoch_count}_tc{self._trial_count}"
        module = optunahub.load_module(package="samplers/auto_sampler")
        # create study
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            storage=f'sqlite:///{os.path.abspath(os.path.join(model_path, "optuna_study.db"))}',
            sampler=module.AutoSampler(),
        )

        print(f"Using sampler: {study.sampler}")

        study.set_user_attr("trial_count", self._trial_count)
        study.set_user_attr("max_file_count", self._max_file_count)
        study.set_user_attr("max_epoch_count", self._max_epoch_count)

        # =============================================================================
        # Optimize
        # =============================================================================
        study.optimize(self.objective, n_trials=self._trial_count)

        # =============================================================================
        # Display results
        # =============================================================================
        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def objective(self, trial: optuna.trial.Trial) -> float:
        """Objective function for Optuna to tune hyperparameters."""

        # fix parameters
        # random_seed = 100
        random_seed = None

        # =============================================================================
        # Define hyperparameters to tune
        # =============================================================================

        model_params: ChessModelParams | None = None
        learning_rate: float | None = None
        batch_size: int | None = None
        train_test_split_ratio: float | None = None
        early_stopping_patience: int | None = None
        early_stopping_threshold: float | None = None
        scheduler_patience: int | None = None
        scheduler_threshold: float | None = None

        # =============================================================================
        # Set the value which won't be tuned
        # =============================================================================

        if True:
            if self._model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D:
                model_params = ChessModelConv2dParams(
                    conv_out_channels=[16, 32, 64], cls_fc_size=256, reg_fc_size=32, cls_head_dropout=0.1, reg_head_dropout=0.5
                )
            elif self._model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_RESNET:
                model_params = ChessModelResnetParams(
                    res_block_sizes=[64], res_block_counts=[1], cls_head_size=256, reg_head_size=64, cls_head_dropout=0.0, reg_head_dropout=0.3
                )
            else:
                raise ValueError(f"Unsupported model name: {self._model_name}")
        learning_rate = 0.0008
        batch_size = 256
        train_test_split_ratio = 0.7
        early_stopping_patience = 10
        early_stopping_threshold = 0.01
        scheduler_patience = 3
        scheduler_threshold = 0.1

        # =============================================================================
        # Configure how each hyperparameter is suggested
        # =============================================================================

        if model_params is None:
            if self._model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D:
                # NOTE: workaround for Optuna not supporting list type directly
                conv_out_channels_categories = {"[16, 32, 64]": [16, 32, 64], "[32, 64, 128]": [32, 64, 128], "[64, 128, 256]": [64, 128, 256]}
                conv_out_channels_name = trial.suggest_categorical("conv_out_channels_option", list(conv_out_channels_categories.keys()))
                conv_out_channels = conv_out_channels_categories[conv_out_channels_name]
                # set model params
                model_params = ChessModelConv2dParams(
                    conv_out_channels=conv_out_channels,
                    cls_fc_size=trial.suggest_categorical("cls_fc_size", [64, 128, 256]),
                    reg_fc_size=trial.suggest_categorical("reg_fc_size", [32, 64, 128]),
                    cls_head_dropout=trial.suggest_categorical("cls_dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                    reg_head_dropout=trial.suggest_categorical("reg_dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                )
            elif self._model_name == ModelUtils.MODEL_NAME.CHESS_MODEL_RESNET:
                # set model params
                model_params = ChessModelResnetParams(
                    # res_block_sizes=HyperParameterTuning.suggest_categorical_listlist_int(trial, "res_block_sizes", [[64]]),
                    res_block_sizes=[64],
                    # res_block_counts=HyperParameterTuning.suggest_categorical_listlist_int(trial, "res_block_counts", [[1]]),
                    res_block_counts=[1],
                    # cls_head_size=trial.suggest_categorical("cls_head_size", [64, 128, 256])
                    cls_head_size=256,
                    # reg_head_size=trial.suggest_categorical("reg_head_size", [32, 64, 128]),
                    reg_head_size=64,
                    cls_head_dropout=trial.suggest_categorical("cls_head_dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                    # cls_head_dropout=0.1,
                    reg_head_dropout=trial.suggest_categorical("reg_head_dropout", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
                    # reg_head_dropout=0.1,
                )
            else:
                raise ValueError(f"Unsupported model name: {self._model_name}")

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
            scheduler_patience = trial.suggest_categorical("scheduler_patience", [2, 3, 4, 5, 7, 10])
        if scheduler_threshold is None:
            scheduler_threshold = trial.suggest_float("scheduler_threshold", 0.001, 0.5, log=True)

        # =============================================================================
        #
        # =============================================================================

        print(model_params)

        # Store the suggested hyperparameters as user attributes for later reference
        trial.set_user_attr("model_params", dataclasses.asdict(model_params))
        trial.set_user_attr("learning_rate", learning_rate)
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("train_test_split_ratio", train_test_split_ratio)
        trial.set_user_attr("early_stopping_patience", early_stopping_patience)
        trial.set_user_attr("early_stopping_threshold", early_stopping_threshold)
        trial.set_user_attr("scheduler_patience", scheduler_patience)
        trial.set_user_attr("scheduler_threshold", scheduler_threshold)
        # add fixed parameters
        trial.set_user_attr("max_file_count", self._max_file_count)
        trial.set_user_attr("max_epoch_count", self._max_epoch_count)
        trial.set_user_attr("model_name", self._model_name)
        trial.set_user_attr("verbose", self._verbose)
        trial.set_user_attr("random_seed", 100)

        # =============================================================================
        # Train the model with the suggested hyperparameters
        # =============================================================================

        model_accuracy_final = TrainCommand.train(
            model_params=model_params,
            batch_size=batch_size,
            learning_rate=learning_rate,
            train_test_split_ratio=train_test_split_ratio,
            early_stopping_patience=early_stopping_patience,
            early_stopping_threshold=early_stopping_threshold,
            scheduler_patience=scheduler_patience,
            scheduler_threshold=scheduler_threshold,
            # fixed parameters
            model_name=self._model_name,
            max_file_count=self._max_file_count,
            max_epoch_count=self._max_epoch_count,
            random_seed=random_seed,
            verbose=self._verbose,
        )

        # Return the final model accuracy as the objective value to maximize
        return model_accuracy_final

    @staticmethod
    def suggest_categorical_listlist_int(trial: optuna.trial.Trial, name: str, choises: list[list[int]]) -> list[int]:
        """workaround for Optuna not supporting list[list] type directly"""
        new_categories = {str(choice): choice for choice in choises}
        choice_name = trial.suggest_categorical(name, list(new_categories.keys()))
        return new_categories[choice_name]


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    # =============================================================================
    # Parse arguments
    # =============================================================================
    argParser = argparse.ArgumentParser(
        description="Perform hyperparameter tuning using Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    argParser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output.")
    argParser.add_argument("-tc", "--trial_count", type=int, default=20, help="Number of trials for hyperparameter tuning.")
    argParser.add_argument("-fc", "--max_file_count", type=int, default=10, help="Maximum number of training files to use.")
    argParser.add_argument("-me", "--max_epoch_count", type=int, default=10, help="Maximum number of epochs for training.")
    argParser.add_argument(
        "--model_name",
        "-mn",
        type=str,
        default=ModelUtils.MODEL_NAME.CHESS_MODEL_CONV2D,
        choices=ModelUtils.get_supported_models(),
        help="Model architecture to use",
    )
    args = argParser.parse_args()

    # =============================================================================
    # Tune hyper parameters
    # =============================================================================
    hyper_parameter_tuning = HyperParameterTuning(
        verbose=args.verbose,
        trial_count=args.trial_count,
        model_name=args.model_name,
        max_file_count=args.max_file_count,
        max_epoch_count=args.max_epoch_count,
    )
    hyper_parameter_tuning.run()
