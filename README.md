## TiTle
- A structured deep learning project template with pytorch-lightning

## Installation
```
source prepare_env.sh
```

## Train

```
python train.py --model=vanilla --dataset=my_dataset --devices=0,1,2,3 model.model_kwargs.hidden_size 1024
```
or
```
python train.py --model=vanilla --dataset=my_dataset1,my_dataset2 --devices=0,1,2,3 model.model_kwargs.hidden_size 1024
```

## Inference/Evaluation

python eval.py --model_dir=logs/vanilla/my_dataset/yyyymmdd-hhmmss_suffix_trained