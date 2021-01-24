# Progressive-GAN
Multi GPU Training Code for GAN with Pytorch

## Ruqeirement 
- Pytorch 1.7.0 +  
  - Nipa 기준 pytorch 1.4 이전 버전 사용시 에러 발생
  
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