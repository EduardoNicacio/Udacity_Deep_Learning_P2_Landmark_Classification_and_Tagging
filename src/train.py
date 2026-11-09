import os
import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot


def _make_grad_scaler(device: torch.device):
    """Create an AMP GradScaler (or None on CPU)."""
    if device.type != "cuda":
        return None

    try:
        # torch >= 2.4 API
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.GradScaler()


def _autocast(device: torch.device):
    """Return an autocast context manager for the given device."""
    try:
        return torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda")
    except (AttributeError, TypeError):  # pragma: no cover - older torch
        return torch.cuda.amp.autocast(enabled=device.type == "cuda")


def train_one_epoch(
    train_dataloader, model, optimizer, loss, scaler=None, device=None
):
    """
    Performs one train_one_epoch epoch.

    Uses automatic mixed precision (AMP) when running on CUDA. Batches that
    trigger a CUDA OOM are skipped (with a warning) so training can continue
    on memory-constrained GPUs.

    :return: the average training loss for the epoch
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if device.type == "cuda":
        model.to(device)
        if scaler is None:
            scaler = _make_grad_scaler(device)

    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to the device
        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad(set_to_none=True)

        try:
            # 2. forward pass (with AMP on CUDA)
            with _autocast(device):
                output = model(data)
                # 3. calculate the loss
                loss_value = loss(output, target)

            # 4. backward pass
            if scaler is not None:
                scaler.scale(loss_value).backward()
                # 5. optimization step (with gradient unscaling)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_value.backward()
                optimizer.step()

        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                print(
                    f"\n[WARNING] CUDA OOM on batch {batch_idx}; skipping batch. "
                    "Consider reducing batch_size."
                )
                optimizer.zero_grad(set_to_none=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            raise

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def _validate(valid_dataloader, model, loss, device=None):
    """Shared validation pass returning (average_loss, accuracy_percent)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    with torch.no_grad():
        model.to(device)
        model.eval()

        valid_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            data, target = data.to(device), target.to(device)

            # 1. forward pass
            output = model(data)
            # 2. calculate the loss
            loss_value = loss(output, target)

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    accuracy = 100.0 * correct / max(total, 1)
    return valid_loss, accuracy


def valid_one_epoch(valid_dataloader, model, loss, device=None):
    """
    Validate at the end of one epoch.

    :return: the average validation loss
    """
    valid_loss, _ = _validate(valid_dataloader, model, loss, device=device)
    return valid_loss


def optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    n_epochs,
    save_path,
    interactive_tracking=False,
    early_stopping_patience=None,
):
    """
    Train the model for ``n_epochs`` epochs, validating after every epoch and
    saving the weights every time the validation loss improves by more than 1%.

    The learning rate is reduced on a plateau of the validation loss
    (``ReduceLROnPlateau``), and AMP is used throughout on CUDA.
    """
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = _make_grad_scaler(device)

    valid_loss_min = None
    epochs_without_improvement = 0
    logs = {}

    # Reduce the learning rate when the validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    # Make sure the directory for the checkpoint exists
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss, scaler=scaler, device=device
        )

        valid_loss, valid_accuracy = _validate(
            data_loaders["valid"], model, loss, device=device
        )

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \t"
            "Validation Accuracy: {:.2f}%".format(
                epoch, train_loss, valid_loss, valid_accuracy
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)

            valid_loss_min = valid_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step(valid_loss)

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()

        # Optional early stopping when the validation loss stops improving
        if (
            early_stopping_patience is not None
            and epochs_without_improvement >= early_stopping_patience
        ):
            print(
                f"Early stopping: validation loss did not improve for "
                f"{early_stopping_patience} epochs."
            )
            break

    return model


def one_epoch_test(test_dataloader, model, loss, device=None):
    """Evaluate the model on the test set and print a loss/accuracy summary."""
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # set the module to evaluation mode
    with torch.no_grad():

        model.to(device)
        model.eval()

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to the device
            data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)
            # 2. calculate the loss
            loss_value = loss(logits, target)

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            pred = logits.argmax(dim=1)

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
