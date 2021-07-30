# PQ Quantization for Weights and Activations

## Installation

Dependencies can be installed with:
`
pip install -r requirements.txt
`

## Quantization

Download the ImageNet1K dataset from https://www.kaggle.com/c/imagenet-object-localization-challenge/data and pass its location as YOUR_IMAGENET_PATH in --data-path argument
For quantizing ResNets `cd` into `src/` and by run the following commands:

- For weight quantization:
```bash
python3 quantize.py --model resnet18 --block-size-cv 9 --block-size-pw 4 --n-centroids-cv 256 --n-centroids-pw 256 --n-centroids-fc 2048 --data-path YOUR_IMAGENET_PATH
```
Final Accuracy : 65.8%

- For activation quantization:
```bash
python3 quantize_act.py --model resnet18 --data-path YOUR_IMAGENET_PATH
```
Final Accuracy on quantizing all ReLUs (except first) : 50%