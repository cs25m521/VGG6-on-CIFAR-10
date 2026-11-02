def evaluate(model, loader, criterion, device):
    model.eval()
    correct = total = 0
    running_loss = 0.0
    with __import__("torch").no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            running_loss += criterion(outputs, targets).item()
            predictions = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += predictions.eq(targets).sum().item()
    return running_loss / len(loader), 100.0 * correct / total

def train_model(model, train_loader, val_loader, optimizer, epochs, criterion, device, run):
    history = {"train_loss": [], "train_accuracy": [], "validation_loss": [], "validation_accuracy": []}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        _, train_acc = evaluate(model, train_loader, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["validation_loss"].append(val_loss)
        history["validation_accuracy"].append(val_acc)
        print(f"Epoch {epoch+1}/{epochs} | train loss {train_loss:.4f} | train acc {train_acc:.2f}% | val loss {val_loss:.4f} | val acc {val_acc:.2f}%")
        run.log({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                 "validation_loss": val_loss, "validation_accuracy": val_acc})
    return history
