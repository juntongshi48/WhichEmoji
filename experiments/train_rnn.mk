SHELL := /bin/bash

train_rnn:
	$(eval GPU_ID:= 7)
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 256)
	$(eval EPOCHS := 20)
	$(eval LR := 0.0005)
	${eval CONFIG_FILE := configs/rnn.yaml}
	-git add .; git commit -m "commit to run"
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python core/main.py --config_file ${CONFIG_FILE} \
		--batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} 

		