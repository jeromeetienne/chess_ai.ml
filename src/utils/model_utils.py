# stdlib imports
import os

# pip imports
import torch

# local imports
from ..libs.chess_model import ChessModel


__dirname__ = os.path.dirname(os.path.abspath(__file__))
data_folder_path = os.path.join(__dirname__, "../../data")

class ModelUtils:
    @staticmethod
    def model_summary(model: torch.nn.Module) -> str:
        """
        Prints a basic summary of the model including parameter count.
        """
        total_params = 0
        trainable_params = 0
        
        output = []
        output.append(f"\nModel Architecture: {model.__class__.__name__}")
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

    @staticmethod
    def save_model(model: ChessModel, folder_path: str):
        # Save the model
        state_dict_path = f"{folder_path}/model.pth"
        torch.save(model.state_dict(), state_dict_path)

    @staticmethod
    def load_model(folder_path: str, input_shape: tuple[int, int, int], output_shape: tuple[int]) -> ChessModel:
        # Load the model
        model = ChessModel(input_shape=input_shape, output_shape=output_shape)
        model_path = f"{folder_path}/model.pth"
        model.load_state_dict(torch.load(model_path))

        return model
