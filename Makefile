# Makefile for MNIST-Min project
IMAGE ?= mnist-min:cpu
APP_DIR ?= mnist_min

DOCKER_RUN = docker run --rm -it -p 7860:7860 -v $(PWD)/$(APP_DIR):/app $(IMAGE)

.PHONY: build shell gradio train onnx tflite ort clean help

help:
	@echo "Targets:"
	@echo "  build   - Build Docker image ($(IMAGE))"
	@echo "  shell   - Open a shell inside the container"
	@echo "  gradio  - Launch Gradio web UI on http://127.0.0.1:7860"
	@echo "  train   - Train model (saves model.pth, class_names.json, history.json)"
	@echo "  onnx    - Export model.onnx from model.pth"
	@echo "  tflite  - Export model.tflite from model.onnx (requires onnx2tf + TF Lite)"
	@echo "  ort     - Run ONNX Runtime inference on samples/0.png"
	@echo "  clean   - Remove artifacts (model.pth, history.json, plots)"

build:
	docker build -t $(IMAGE) $(APP_DIR)

shell:
	$(DOCKER_RUN) bash

gradio:
	$(DOCKER_RUN) python app.py --server.port 7860 --server.host 0.0.0.0

train:
	$(DOCKER_RUN) python train.py --epochs 2 --batch-size 128 --save-plots

onnx:
	$(DOCKER_RUN) bash -lc "python export_onnx.py --model model.pth --out model.onnx"

tflite:
	$(DOCKER_RUN) bash -lc "python export_tflite.py --onnx model.onnx --out model.tflite --quant int8"

ort:
	$(DOCKER_RUN) bash -lc "python make_samples.py --count 1 && python onnxruntime_infer.py --onnx model.onnx --image samples/0.png"

clean:
	rm -f $(APP_DIR)/model.pth $(APP_DIR)/model.onnx $(APP_DIR)/model.tflite \
	      $(APP_DIR)/history.json $(APP_DIR)/loss.png $(APP_DIR)/acc.png
