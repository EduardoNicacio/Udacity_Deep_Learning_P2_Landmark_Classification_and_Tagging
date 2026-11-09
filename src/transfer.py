import torch
import torchvision
import torchvision.models as models
import torch.nn as nn


def _replace_classifier(model: nn.Module, n_classes: int) -> nn.Module:
    """
    Replace the last fully-connected layer of ``model`` with a new
    ``nn.Linear`` mapping to ``n_classes`` outputs.

    This works for architectures that expose their head as ``model.fc``
    (ResNet), ``model.classifier[1]`` (EfficientNet) or
    ``model.classifier[3]`` (MobileNetV3), as well as any other module that
    ends in a ``nn.Linear`` layer.
    """
    last_linear_name = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_linear_name = name

    if last_linear_name is None:
        raise ValueError(
            f"Could not find a linear classification head in {type(model).__name__}"
        )

    parent_path, _, attr_name = last_linear_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    old_layer = getattr(parent, attr_name)

    setattr(parent, attr_name, nn.Linear(old_layer.in_features, n_classes))
    return model


def get_model_transfer_learning(model_name="resnet18", n_classes=50):
    """
    Build a pretrained model with a frozen backbone and a fresh classifier
    head with ``n_classes`` outputs.

    :param model_name: any classification architecture known to
        ``torchvision.models`` (e.g. ``resnet18``, ``resnet50``,
        ``efficientnet_b0``, ``mobilenet_v3_small``)
    :param n_classes: number of output classes
    :return: the modified model (backbone frozen, head trainable)
    """

    # Get the requested architecture, with pretrained ImageNet weights
    try:
        weights = models.get_model_weights(model_name).DEFAULT
        model_transfer = models.get_model(model_name, weights=weights)
    except ValueError:
        torchvision_major_minor = ".".join(torchvision.__version__.split(".")[:2])

        raise ValueError(
            f"Model {model_name} is not known. List of available models: "
            f"https://pytorch.org/vision/{torchvision_major_minor}/models.html"
        )

    # Freeze all parameters in the model
    for param in model_transfer.parameters():
        param.requires_grad = False

    # Add the linear layer at the end with the appropriate number of classes.
    # The newly created nn.Linear has requires_grad=True by default, so only
    # the head is trained while the backbone stays frozen.
    model_transfer = _replace_classifier(model_transfer, n_classes)

    return model_transfer


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_get_model_transfer_learning(data_loaders):

    model = get_model_transfer_learning(n_classes=23)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
