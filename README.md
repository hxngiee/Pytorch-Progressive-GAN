# Progressive-GAN
Multi GPU Training Code for GAN with Pytorch

## Ruqeirement 
- Pytorch 1.7.0 +  
  - Nipa 기준 pytorch 1.2, 1.4버전 에러 발견
  
## Usage
- main.py, main_single_gpu.py를 분리해서 사용할 것 
  - main.py에서 train_single_gpu, train_data_parallel을 import하거나
  - main_single_gpu에서 train_dist_parallel를 import해서 사용하지 말 것
    - 인자개수가 맞지 않음

### single gpu
```
# main : main_single_gpu.py / train : train_single_gpu.py
python main_single_gpu.py 
```

### DataParallel
```
# main : main_single_gpu.py / train : train_data_parallel.py
python main_single_gpu.py
```

### DistributedDataParallel
```
# main : main.py / train : train_dist_parallel.py
python main.py --gpu_device 0 1 2 3 --batch_size 768
```