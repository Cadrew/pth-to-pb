# PTH to PB

⚠️ This code is a draft for testing and exploration ⚠️

Converting ESRGAN PyTorch models to SavedModel files for Tensorflow.

# Prerequisites

Install:
```bash
pip install onnx
pip install onnx_tf
pip install torch
pip install basicsr
pip install git+https://github.com/sberbank-ai/Real-ESRGAN.git
```

# Usage

## Enhance an image

`run_model_pytorch` comes from an implementation of Real-ESRGAN with PyTorch [here](https://github.com/ai-forever/Real-ESRGAN).

To enhance an image:
```bash
python .\run_model_pytorch.py
```

## Convert a model

`conversion.py` is specifically designed to use `basicsr.RRDBNet` model in PyTorch to convert ESRGAN models into Tensorflow models.

To run a conversion:
```bash
python .\conversion.py
```