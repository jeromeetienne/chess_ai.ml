class EarlyStopper:
    """
    Earlt stopper for pyTorch training loops.
    """

    def __init__(self, patience: int = 10, threshold: float = 0.001):
        """
        Early stops the training if validation loss doesn't improve after a given patience.

        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            threshold (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.threshold = threshold
        self.bad_epoch_count = 0
        self.min_validation_loss = float("inf")

    def step(self, validation_loss: float) -> tuple[bool, bool]:
        """
        Checks if training should be stopped early based on validation loss.

        Args:
            - validation_loss (float): Current validation loss.

        Returns:
            - (bool, bool): A tuple where the first element indicates whether to stop training,
                            and the second element indicates whether to save the model.
        """
        must_stop = False
        must_save = False

        # Check if the validation loss has improved by at least min_delta
        is_improved = validation_loss < (self.min_validation_loss - self.threshold)

        if is_improved:
            # If improved, update the minimum validation loss and reset wait counter
            self.min_validation_loss = validation_loss
            self.bad_epoch_count = 0
            must_save = True  # Signal to save the model
        else:
            # If not improved, increment the wait counter
            self.bad_epoch_count += 1
            if self.bad_epoch_count >= self.patience:
                # If patience exceeded, signal to stop training
                must_stop = True

        return (must_stop, must_save)
