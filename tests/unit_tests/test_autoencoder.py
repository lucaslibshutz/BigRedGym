import pytest
import torch

from learning.modules.autoencoder import Autoencoder

NUM_CHANNELS = 3
WINDOW_LENGTH = 51
LATENT_DIM = 8

@pytest.fixture
def ae():
    torch.manual_seed(0)
    return Autoencoder(NUM_CHANNELS, WINDOW_LENGTH, LATENT_DIM)

@pytest.mark.parametrize("batch_size",[1,16])
def test_shapes(ae, batch_size):
    x = torch.randn(batch_size, NUM_CHANNELS, WINDOW_LENGTH)
    x_hat, z = ae(x)
    assert x_hat.shape == x.shape
    assert z.shape == (batch_size, LATENT_DIM)

def test_samples_are_indpendent(ae):
    x = torch.randn(16, NUM_CHANNELS, WINDOW_LENGTH)
    x_hat, _ = ae(x)
    x_hat_single, _ = ae(x[3:4])
    torch.testing.assert_close(x_hat[3:4], x_hat_single)


def test_gradient_reaches_every_parameter(ae):
    ae.loss(torch.randn(16, NUM_CHANNELS, WINDOW_LENGTH)).backward()
    for name, param in ae.named_parameters():
        assert param.grad is not None, name
        assert param.grad.abs().sum() >0, name
