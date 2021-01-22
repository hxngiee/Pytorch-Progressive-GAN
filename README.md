# Progressive-GAN
Multi GPU Training Code for GAN with Pytorch
#### To do
- Image Generation 프레임워크 DCGAN에 맞게 세팅
  - 모델 저장 및 로드-> utils에서 save, load 모델에 맞게 수정할 것, test image generation, tensorboardX 적용
- train_dist_parallel에 eps 반영

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

## Tips

## Error

```
RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicat
es that your module has parameters that were not used in producing loss. You can enable unused parameter detection by 
(1) passing the keyword argument `find_unused_parameters=True` to `torch.nn.parallel.DistributedDataParallel`; (2) mak
ing sure all `forward` function outputs participate in calculating loss. If you already have done the above two steps,
 then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module
's `forward` function. Please include the loss function and the structure of the return value of `forward` of your mod
ule when reporting this issue (e.g. list, dict, iterable)
``` 

- DistributedDataParallel train시 eps(gradient penalty) 파트가 아직 반영 안됨
  - 원인 : eps 파트가 forward 부분이랑 연결이 안됨
    -  train_dist_parallel 파트에선 임시로 지워둔 상태
  - [해결책](https://study-grow.tistory.com/entry/pytorch-%EC%97%90%EB%9F%AC-DistributedDataParallel-%EC%97%90%EB%9F%AC) 
    

```
Traceback (most recent call last):
  File "/home/ubuntu/anaconda3/envs/pggan/lib/python3.8/site-packages/torch/multiprocessing/spawn.py", line 19, in _wrap
    fn(i, *args)
  File "/home/ubuntu/hongiee/pggan_multi/train_dist_parallel.py", line 206, in train
    optimG, optimD, st_epoch = load(ckpt_dir=ckpt_dir,
  File "/home/ubuntu/hongiee/pggan_multi/util.py", line 84, in load
    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)
IndexError: list index out of range
```
- main.py에서 checkpoint 폴더에 모델이 없는데  train_continue가 on으로 설정되어 있으면 생기는 에러
  - 초기 학습시 train_continue를 off로 설정하고 모델이 저장되면 on으로 학습할 것