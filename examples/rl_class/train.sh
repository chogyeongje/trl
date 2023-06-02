CCELERATE_LOG_LEVEL=info accelerate launch rl_class.py \
	--batch_size 8 \
	--mini_batch_size 8 \
	--use_usefulness True \
	--use_harmfulness True 
