import torch


class ModelSummary:
    @staticmethod
    def print(model: torch.nn.Module) -> None:
        output_str = ModelSummary.to_string(model)
        print(output_str)

    @staticmethod
    def to_string(model: torch.nn.Module) -> str:
        """
        Prints a basic summary of the model including parameter count.
        """
        total_params = 0
        trainable_params = 0

        output = []
        output.append(f"Model Architecture: {model.__class__.__name__}")
        output.append("-" * 60)
        output.append(f"{'Layer Name':<30} {'Param Count':>15} {'Trainable':>10}")
        output.append("=" * 60)

        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue

            param = parameter.numel()
            total_params += param

            if parameter.requires_grad:
                trainable_params += param

            output.append(f"{name:<30} {param:>15,} {'Yes' if parameter.requires_grad else 'No':>10}")

        output.append("=" * 60)
        output.append(f"Total Parameters: {total_params:,}")
        output.append(f"Trainable Parameters: {trainable_params:,}")
        output.append(f"Non-trainable Parameters: {total_params - trainable_params:,}")
        output.append("-" * 60)

        return "\n".join(output)
