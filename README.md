# Progressive-GAN
Multi GPU Training Code for GAN with Pytorch
#### To do
- Image Generation 프레임워크 DCGAN에 맞게 세팅
  - 없는 폴더 생성, 모델 저장 및 로드, model generation test
- train_dist_parallel에 eps 반영

## Ruqeirement 
- Pytorch 1.7.0 +  
> Nipa 기준 pytorch 1.2, 1.4 에러 발견
  
## Usage
### single gpu
```
# main_single_gpu, train_single_gpu
python main_single_gpu.py 
```

### DataParallel
```
# main_single_gpu, train_data_parallel
python main_single_gpu.py
```

### DistributedDataParallel
```
# main.py, train_dist_parallel
python main.py --gpu_device 0 1 2 3 --batch_size 768
```

## Error
- DistributedDataParallel train시 eps(gradient penalty) 파트가 아직 반영 안됨
    - 원인 : eps 파트가 forward 부분이랑 연결이 안됨
      -  train_dist_parallel 파트에선 임시로 지워둔 상태
```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicat
es that your module has parameters that were not used in producing loss. You can enable unused parameter detection by 
(1) passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`; (2) mak
ing sure all `forward` function outputs participate in calculating loss. If you already have done the above two steps,
 then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module
's `forward` function. Please include the loss function and the structure of the return value of `forward` of your mod
ule when reporting this issue (e.g. list, dict, iterable)
```

    - [해결책](https://study-grow.tistory.com/entry/pytorch-%EC%97%90%EB%9F%AC-DistributedDataParallel-%EC%97%90%EB%9F%AC) 
