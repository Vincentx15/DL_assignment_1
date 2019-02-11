import torch


def train(model, train_loader, loss_fn, optimizer, epoch, device, tb_writer):
    """Train one epoch for model."""
    model.train()

    running_loss = 0.0
    for batch_idx, (inputs, _, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        pred = model(inputs)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(inputs), len(train_loader)*len(inputs),
                100. * batch_idx / len(train_loader), loss.item()))
            # tensorboard logging
            tb_writer.add_scalar("Training loss", loss.item(),
                                 epoch * len(train_loader) + batch_idx)

    return running_loss/len(train_loader.dataset)


def test(model, test_loader, test_loss_fn, device):
    """ Compute accuracy and loss of model over given dataset. """
    model.eval()

    test_loss = 0
    correct = 0
    test_size = 0
    with torch.no_grad():
        for inputs, _, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)

            output = model(inputs)
            test_size += len(inputs)
            test_loss += test_loss_fn(output, target).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_size
    accuracy = correct / test_size
    print('\nTest set: Average loss: {:.4f},\
           Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, test_size,
          100. * accuracy))

    return test_loss, accuracy

