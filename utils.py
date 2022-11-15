import torch
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import engine

default_device = "cuda" if torch.cuda.is_available() else "cpu"


def save_model(
        model: torch.nn.Module,
        path: str,   
    ) -> None:
    """Saves a PyTorch model to a target directory.
    
    Args:
      model: A target PyTorch model to save.
      target_dir: A directory for saving the model to.
      model_name: A filename for the saved model. Should include
        either ".pth" or ".pt" as the file extension.
    
    Example usage:
      save_model(model=model_0,
                 target_dir="models",
                 model_name="05_going_modular_tingvgg_model.pth")
    """
    
    assert path.endswith(".pth") or path.endswith(".pt")

    
    abs_path = os.path.abspath(path)
    dirname = os.path.dirname(abs_path)
    os.makedirs(dirname, exist_ok=True)
    
    print(f"Saving model to: {path}")
    torch.save(obj=model.state_dict(), f=path)
    
def plot_loss_curves(results: Dict[str, List[float]]) -> None:
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    loss = results['train_loss']
    test_loss = results['test_loss']
    
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    
    # Figure out how many epochs
    epochs = range(len(loss))
    
    plt.figure(figsize=(15,7))
    
    # plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    # plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.show()
    
def create_writer(model_name: str) -> torch.utils.tensorboard.writer.SummaryWriter:
    from datetime import datetime
    from torch.utils.tensorboard import SummaryWriter
    import os
    
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    log_dir = os.path.join("./runs", timestamp, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
    
def train_and_save(
    # TODO: parameters
    model: torch.nn.Module,
    train_dataloader,
    test_dataloader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
    model_name: str,
    model_dir: str = "./models",
    device: torch.device = default_device
) -> Dict[str, List]:
    
    model = model.to(device)
    
    writer=create_writer(model_name)
    
    model_path = os.path.join(model_dir, model_name) + '.pt'
    
    metrics=engine.train(model=model,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=epochs,
                writer=writer,
                device=device)
    
    
    save_model(model=model, path=model_path)
    
    plot_loss_curves(metrics)

    return metrics

def set_seeds(seed: int=42):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    
    
def summary(model, input_size):
    import torchinfo

    return torchinfo.summary(model,
        input_size=input_size,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
    )
