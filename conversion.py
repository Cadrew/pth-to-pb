import torch
import torch.onnx
import onnx
import onnx_tf
from basicsr.archs.rrdbnet_arch import RRDBNet

# An instance of your model
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
model.load_state_dict(torch.load('weights/RealESRGAN_x4plus.pth')['params_ema'])
# model.load_state_dict(torch.load('weights/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth')['params'])

# set the train mode to false since we will only run the forward pass.
model.train(False)
model.cpu().eval()

# An example input you would normally provide to your model's forward() method
x = torch.rand(1, 3, 64, 64)

# Export the model
with torch.no_grad():
    torch_out = torch.onnx._export(model, x, 'onnx/realesrgan-x4.onnx', opset_version=11, export_params=True)

# Load the ONNX model
onnx_model = onnx.load('onnx/realesrgan-x4.onnx')

# Convert the ONNX model to Tensorflow SavedModel
tf_model = onnx_tf.backend.prepare(onnx_model)
tf_model.export_graph('SavedModel/saved_model')