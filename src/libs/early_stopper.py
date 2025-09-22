class EarlyStopper:
    def __init__(self, patience:int=1, min_delta:float=0):
        """
        Early stops the training if validation loss doesn't improve after a given patience.

        Args:
            patience (int): How many epochs to wait after last time validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss: float) -> tuple[bool, bool]:
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
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            must_save = True
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                must_stop = True

        return (must_stop, must_save)