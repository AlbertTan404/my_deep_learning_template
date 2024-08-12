## TiTle
- A structured deep learning project template with pytorch-lightning

## Installation
```
source prepare_env.sh
```

## Train

```
python train.py --model=vanilla --dataset=my_dataset --devices=0,1,2,3
```
or

~~``` python train.py --model=vanilla --dataset=my_dataset --devices=0,1,2,3 model.model_kwargs.hidden_size=1024 model.training_kwargs.lr=1e-5 ```~~

```
python train.py --model=vanilla --dataset=my_dataset --devices=0,1,2,3 hidden_size=1024 lr=1e-5  # it will automatically search the key now! (but mind that duplicated key in different sub-configs is not supported)
```


## Inference/Evaluation

```
python eval.py --model_dir=logs/vanilla/my_dataset/yyyymmdd-hhmmss_suffix_trained/ckpt_name.ckpt --device=1
```
