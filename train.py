def train_model(model, train_loader, criterion, optimizer):
    num_epochs = 50
    best_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.long().squeeze())  # Convert to long and remove extra dimension
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
        if epoch % 5 == 0:
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

        # Early stopping
        if round(epoch_loss, 4) < round(best_loss, 4):
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered")
                break