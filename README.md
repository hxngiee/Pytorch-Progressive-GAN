# Progressive-GAN
Multi GPU Training Code for GAN

## Ruqeirement 
- Pytorch 1.7.0 +  
  
## Train
```
python main.py --mode train_single 
python main.py --gpu_device 0 1 2 --mode train_multi --batch_size 512
```

## Test
```
python main.py --mode test
```

## ISSUE
```
# backward시 grad_fn에 AddBackward가 아닌 GatherBackward가 생김 
grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat,create_graph=True)[0]
grad_x_hat.backward()
```
* [[link]](https://discuss.pytorch.org/t/gradient-penalty-in-wgan-gp-not-converging-on-multi-gpu/35528) `gradient penalty` in WGAN-GP not support on Multi-GPU training
