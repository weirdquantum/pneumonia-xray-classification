import pytest
import torch

from pneumonia.gradcam import GradCAM, border_share
from pneumonia.models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_and_gradcam_shapes(name):
    torch.manual_seed(0)
    spec = build_model(name, pretrained=False)
    model = spec.model.eval()
    x = torch.randn(2, 3, 224, 224)
    assert model(x).shape == (2, 1)

    cams, probs = GradCAM(model, spec.cam_layer, spec.cam_reshape)(x)
    assert cams.shape == (2, 224, 224)
    assert probs.shape == (2,)
    assert torch.isfinite(cams).all()
    assert cams.min() >= 0 and cams.max() <= 1


def test_border_share_of_uniform_map():
    share = border_share(torch.ones(1, 224, 224))
    assert share.item() == pytest.approx(1 - 0.75**2, abs=0.01)
