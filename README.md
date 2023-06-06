# README

## Install

```
pip install -e .  # trl 설치
```

## Add Dataset

아래 경로에 사용할 train, test 데이터셋을 추가

```
train_path: trl/examples/rl_class/datasets/train
test_apth: trl/examples/rl_class/datasets/test
```



## Train Model

```
cd trl/examples/rl_class
bash train.sh				# 필요시 train.sh 수정
```



### Training Arguments

+ model_name[str]: 사용할 모델 명 (default: microsoft/DialoGPT-medium)
+ learning_rate[float]: 모델의 learning rate (default: 1.47e-5)
+ batch_size[int]: 전체 gpu 에서 한번에 처리하는 batch 크기 (default=16)
  + reward, constraint 를 계산하는 데이터 batch 크기
+ mini_batch_size[int]: 계산된 reward, constraint 로 모델을 학습할 때 사용될 batch 크기 (default=4)
  + ppo 를 사용하여 사용될 때 사용되는 데이터 batch 크기
+ ppo_epoch[int]: ppo 학습을 진행할 epoch 크기 (default=5)
+ use_usefulness[bool]: usefulness 모델을 사용하여 reward 계산 하여 학습에 사용할지의 유무 (default=True)
  + False 인 경우 reward 값은 항상 0을 사용
+ use_harmfulness[bool]: harmfulness 모델을 사용하여 constraint 계산 하여 학습에 사용할지의 유무 (default=True)
  + use_harmfulness 가 True 인 경우에만, 아래 lambda 옵션들을 사용함
  + False 인 경우 constraint 값은 항상 0을 사용
+ lambda_type[str]: 사용할 lambda 모델 (default=constant)
  + ["constant", "linear"] 중 하나 선택
  + "constant": 상수값의 lambda 사용
  + "linear": harmfulness 모델 input data 의 embedding 값을 linear 연산 한 뒤, 평균값을 계산해 lambda 로 사용
+ lambda_value[float]: lambda_type 이 constant 일 시, 사용할 lambda 값 (default=-1)
  + -1 로 설정될 경우, 랜덤으로 초기화된 값을 사용
+ lambda_lr[float]: lambda 모델의 학습에 사용될 learning_rate (default=0.1)
  + 0 이하의 값으로 설정될 경우, lambda 모델의 학습이 진행되지 않음
+ max_constraint[float]: constraint 의 상한선으로 사용될 값 (default=1)
+ model_save_path[str]: 모델을 저장할 위치 (default=ms_dialogpt_medium)

### Experiments Setting

실험에 따라 아래와 같이 argument 설정

```
# 공통 (상황에 따라 적절히 값 수정 가능)
--batch_size 64 \
--mini_batch_size 16 \
--ppo_epoch 5 \
--model_save_path test \

# Exp 3
--use_usefulness True \
--use_harmfulenss False

# Exp 4
--use_usefulness True \
--use_harmfulness True \
--lambda_type constant \
--lambda_value 0.1 \		# 원하는 값으로 수정
--lambda_lr 0 \
--max_constraint 1			# 원하는 값으로 수정

# Exp 5
--use_usefulness True \
--use_harmfulness True \
--lambda_type constant \
--lambda_value -1 \		
--lambda_lr 0.1 \			# 원하는 값으로 수정
--max_constraint 1			# 원하는 값으로 수정

# Exp 6
--use_usefulness True \
--use_harmfulness True \
--lambda_type linear \
--lambda_value 0.1 \
--lambda_lr 0.1	\			# 원하는 값으로 수정
--max_constraint 1			# 원하는 값으로 수정
```

