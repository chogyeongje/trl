CCELERATE_LOG_LEVEL=info accelerate launch rl_class.py \
	--batch_size 8 \
	--mini_batch_size 8 \
	--ppo_epoch 1 \
	--model_save_path test \
	--use_usefulness True \
	--use_harmfulness True \
	--lambda_type constant \
	--lambda_value -1 \
	--max_constraint 1.0 \
	--lambda_lr 0.01	


# # Exp 3
# --use_usefulness True
# --use_harmfulness False
# 
# # Exp 4
# --use_usefulness True
# --use_harmfulness True
# --lambda_type constant
# --lambda_value 0.1
# --lambda_lar = 0
# 
# # Exp 5
# --use_usefulness True
# --use_harmfulness True
# --lambda_type constant
# --lambda_value -1
# 
# # Exp 6
# --use_usefulness True
# --use_harmfulness True
# --lambda_type linear
# --lambda_value -1
