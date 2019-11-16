

.PHONY: download transform train tune model calibrate evaluate qa release

base_dir ?= '.'
version ?= latest  # git rev-parse --short HEAD

build:
	docker build --build-arg APP_VERSION=${version} -t suneetamall/e2e-ml-on-k8s:${version} .

push:
	docker push suneetamall/e2e-ml-on-k8s:$(version)

download:
	python app/download_petset.py --output $(base_dir)/out

transform:
	python app/dataset_gen.py --input $(base_dir)/warehouse --output $(base_dir)/out

train:
	python app/train.py --model_arch VGG19 --input $(base_dir)/transform --output $(base_dir)/out \
	--checkpoint_path $(base_dir)/out/ckpts --tensorboard_path $(base_dir)/out

tune:
	python app/tune.py --num_samples 4 --input $(base_dir)/transform --output $(base_dir)/out

model:
	python app/train.py --model_arch VGG19 --input $(base_dir)/transform \
	--hyperparam_fn_path $(base_dir)/tune/optimal_hp.json --output $(base_dir)/out \
	--checkpoint_path $(base_dir)/out/ckpts --tensorboard_path $(base_dir)/out;
	mkdir -p $(base_dir)/out/resource/;
	ln -s $(base_dir)/model/model.h5 $(base_dir)/out/resource/model.h5;
	ln -s $(base_dir)/tune/optimal_hp.json $(base_dir)/out/resource/model_params.json;

calibrate:
	python app/calibrate.py --input $(base_dir)/transform --model_weight $(base_dir)/model/resource/model.h5 \
	--output $(base_dir)/out;
	ln -s $(base_dir)/model/resource/model.h5 $(base_dir)/out/model.h5;
	ln -s $(base_dir)/model/resource/model_params.json $(base_dir)/out/model_params.json;

evaluate:
	papermill evaluate.ipynb $(base_dir)/out/Report.ipynb \
	-p model_weights $(base_dir)/calibrate/model.h5 \
	-p calibration_weights $(base_dir)/calibrate/calibration.weights \
	-p input_data_dir $(base_dir)/transform \
	-p hyperparameters $(base_dir)/calibrate/model_params.json

qa:
	python app/test.py --input $(base_dir)/tf-data/training/Abyssinian_2/image.jpg --output $(base_dir)/out/result.jpg

release:
	echo release

all:
	echo release



