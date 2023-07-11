import torch
import torch_tensorrt

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
import timm

# load model
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).eval().to("cuda")
model = timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k")
traced_model = torch.jit.script(model).save("dummy_vit.pt")

# Compile with Torch TensorRT;
# trt_model = torch_tensorrt.compile(model,
#     inputs= [torch_tensorrt.Input((1, 3, 224, 224))],
#     enabled_precisions= { torch.half} # Run with FP32
# )

# # Save the model
# torch.jit.save(trt_model, "model.pt")
