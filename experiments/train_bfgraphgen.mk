SHELL := /bin/bash


train_basic:
	$(eval GPU_ID:= 2 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/bfndiscrete.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	_code=${REVISION}_$${RANDOM}; CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
		--exp_name ${EXP_NAME}_$${_code} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}



train_baseline_1:
	$(eval GPU_ID:= 0 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/bfn4graphgen.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
		--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}

## Comm-20
train_bfn_comm-20:
	$(eval GPU_ID:= 6 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 256)
	$(eval EPOCHS := 100000)
	${eval CONFIG_FILE := configs/bfn_comm-20.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
			--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}
## Planar
train_bfn_planar:
	$(eval GPU_ID:= 7 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 64)
	$(eval EPOCHS := 20000)
	${eval CONFIG_FILE := configs/bfn_planar.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
			--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}
## BFN
## SBM
train_bfn_sbm:
	$(eval GPU_ID:= 5 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 4)
	$(eval EPOCHS := 10000)
	${eval CONFIG_FILE := configs/bfn_sbm.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
			--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}
## BFN
train_bfn_qm9withh:
	$(eval GPU_ID:= 7 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 256)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/bfn_qm9withh.yaml}
	-git add .; git commit -m "commit to run"
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
		--exp_name ${EXP_NAME}_${REVISION}_n$${beta1}_e$${beta2} --batch_size ${BATCH_SIZE} \
		--epochs ${EPOCHS} --beta_node $${beta1} --beta_edge 


train_tune1k_b1_b2:
	$(eval GPU_ID:= 0,1,2,3,4,5,6,7 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 256)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/bfn_qm9withh.yaml}
	-git add .; git commit -m "commit to run"
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	for beta1 in 5 4 3 2 1; do \
		for beta2 in 5 4 3 2 1; do \
			echo python src/main.py  \
					--config_file ${CONFIG_FILE}  \
					--exp_name ${EXP_NAME}_${REVISION}_n$${beta1}_e$${beta2} --batch_size ${BATCH_SIZE} \
					--epochs ${EPOCHS} --beta_node $${beta1} --beta_edge $${beta2}; \
		done; \
	done | shuf | python experiments/parallel_schedule_stdin.py --resources ${GPU_ID}

tune1k_bn_leq_be_equal_lambda:
	$(eval GPU_ID:= 0,1,2,3,4,5,6,7)
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 256)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/bfn_qm9withh.yaml}
	-git add .; git commit -m "commit to run"
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	for beta1 in 5 4 3 2 1; do \
		for beta2 in 5 4 3 2 1; do \
			if ((beta1 < beta2)); then\
				echo python src/main.py  \
						--config_file ${CONFIG_FILE}  \
						--exp_name ${EXP_NAME}_${REVISION}_n$${beta1}_e$${beta2} --batch_size ${BATCH_SIZE} \
						--epochs ${EPOCHS} --beta_node $${beta1} --beta_edge $${beta2}; \
			fi\
		done; \
	done | shuf | python experiments/parallel_schedule_stdin.py --resources ${GPU_ID}


test_bfn_qm9withh:
	$(eval GPU_ID:= 5 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/bfn_qm9withh-test.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
		--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}

## Dummy
train_bfn_qm9withh_dummy:
	$(eval GPU_ID:= 7 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/bfn_qm9withh_dummy.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
		--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}

train_digress_qm9withh_dummy:
	$(eval GPU_ID:= 7 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/digress_qm9withh_dummy.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
		--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}


## Digress

test_digress_qm9withh:
	$(eval GPU_ID:= 6 )
	$(eval EXP_NAME:= $@) 
	$(eval BATCH_SIZE := 512)
	$(eval EPOCHS := 1000)
	${eval CONFIG_FILE := configs/digressqm9withhtestonly.yaml}
	${eval REVISION := ${shell git rev-parse --short HEAD}}
	CUDA_VISIBLE_DEVICES=${GPU_ID} python src/main.py --config_file ${CONFIG_FILE} \
		--exp_name ${EXP_NAME} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS}