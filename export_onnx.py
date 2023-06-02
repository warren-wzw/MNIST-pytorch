#!/usr/bin/env python3
import torch
from simple_net import SimpleModel


# Load the pretrained model and export it as onnx
model = SimpleModel()
model.eval()
checkpoint = torch.load("weight.pth", map_location="cpu")
model.load_state_dict(checkpoint)

# Prepare input tensor
input = torch.randn(1, 1, 28, 28, requires_grad=True)#batch size-1 input cahnne-1 image size 28*28

# Export the torch model as onnx
torch.onnx.export(model,
            input,
            'model.onnx', # name of the exported onnx model
            opset_version=13,
            export_params=True,
            do_constant_folding=True)
