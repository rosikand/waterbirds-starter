# Waterbirds starter code 

Provides scaffolding for training models on the [Waterbirds dataset](https://github.com/kohpangwei/group_DRO) for robustness techniques. Makes use of [torchplate](https://github.com/rosikand/torchplate). 

**Directions**: 

1. Specify your config parameters (e.g., data paths, batch size) in `configs.py`. 
2. Create custom experiments in `experiments.py` to test out new methods or use `experiments.BaseExperiment`. 
3. Train and test the method using `runner.py` and specify the relevant CLI args. Example: 

```
python runner.py -e BaseExp -c BaseConfig -train -test -num_epochs 10 -grad_accum 1
```
