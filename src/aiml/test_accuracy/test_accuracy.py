def test_accuracy(model, dataloader, device):
    """This function returns the accuracy of a given dataset on a pre-trained model."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            outputs = model(x)
            _, predictions = torch.max(outputs, 1)
            predictions = predictions.to('cpu')
            total += y.size(0)
            correct += (predictions == y).sum().item()
    accuracy = correct / total
    return accuracy
