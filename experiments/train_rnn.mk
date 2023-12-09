SHELL := /bin/bash

train_rnn:
	$(eval GPU_ID:= 7)
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 1024)
	$(eval EPOCHS := 10)
	$(eval LR := 0.0005)
	${eval CONFIG_FILE := configs/rnn.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python core/main.py --config_file ${CONFIG_FILE} \
		--batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} 

		