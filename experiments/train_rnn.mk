SHELL := /bin/bash

train_rnn:
	$(eval GPU_ID:= 4)
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 30)
	$(eval LR := 0.0005)
	${eval CONFIG_FILE := configs/rnn.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python core/main.py --config_file ${CONFIG_FILE} \
		--batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} 

train_rnn_multiclass:
	$(eval GPU_ID:= 4)
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 80)
	$(eval LR := 0.0005)
	$(eval NUM_OUTPUT_LABELS := 2)
	${eval CONFIG_FILE := configs/rnn.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python core/main.py --config_file ${CONFIG_FILE} \
		--batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --lr ${LR} \
		--num_output_labels ${NUM_OUTPUT_LABELS}