import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from config import SEED, BATCH_SIZE, LEARNING_RATE, EPOCHS, PROJECT, RUN_NAME
from data import get_cifar10, set_seed
from model import build_model
from engine import train_model, evaluate

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    train_loader, val_loader, test_loader = get_cifar10(BATCH_SIZE)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    run = wandb.init(project=PROJECT, name=RUN_NAME, config={
        "seed": SEED, "architecture": "VGG6", "activation": "GELU",
        "optimizer": "Adam", "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "batch_norm": True
    })
    history = train_model(model, train_loader, val_loader, optimizer,
                          EPOCHS, criterion, device, run)
    best_val = max(history["validation_accuracy"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    run.summary["best_validation_accuracy"] = best_val
    run.summary["test_loss"] = test_loss
    run.summary["test_accuracy"] = test_acc
    print(f"Best validation accuracy: {best_val:.2f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test top-1 accuracy: {test_acc:.2f}%")
    torch.save(model.state_dict(), "vgg6_gelu_adam.pth")
    print("Saved model: vgg6_gelu_adam.pth")
    wandb.finish()

if __name__ == "__main__":
    main()
