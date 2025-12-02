import pytorch_lightning as pl
import torch

class TestModel(pl.LightningModule):
    def __init__(self):
        super(TestModel, self).__init__()
        self.layer = torch.nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def test_forward_pass(self):
        model = TestModel()
        x = torch.randn(1, 10)
        output = model(x)
        assert output.shape == (1, 1)

    def test_layer_weights(self):
        model = TestModel()
        assert model.layer.weight.shape == (1, 10)