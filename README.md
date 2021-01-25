# Progressive-GAN
Multi GPU Training Code for GAN with Pytorch

## Ruqeirement 
- Pytorch 1.7.0 +  
  - pytorch 1.4 이전 버전 사용시 에러 발생
  
## Usage
### single gpu
```
python main.py --mode train_single --train_continue off
python main.py --mode train_single --train_continue on
```
### multi gpu
```
python main.py --gpu_device 0 1 2 --mode train_multi --train_continue off
python main.py --gpu_device 0 1 2 --mode train_multi --train_continue on
```
### test
```
python main.py --mode test
```

## Troubleshooting
Multi GPU를 이용한 DDP에서 `gradient penalty`에 문제가 있어 반영 안한 상태 - [링크](https://discuss.pytorch.org/t/gradient-penalty-in-wgan-gp-not-converging-on-multi-gpu/35528)
```
# backward시 grad_fn에 AddBackward가 아닌 GatherBackward가 생김 
grad_x_hat = grad(outputs=hat_predict.sum(), inputs=x_hat,create_graph=True)[0]
grad_x_hat.backward()
```
