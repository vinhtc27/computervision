PYTHON ?= .venv/bin/python
SCRIPT = bow.py
ARGS   ?=

.PHONY: help install run baseline optimal clean clean-final

help:
	@echo ""
	@echo "Cách dùng: make <target> [ARGS='--feature sift --vocab-size 500 --classifier svm_rbf']"
	@echo ""
	@echo "  Setup"
	@echo "    install      Cài dependencies vào .venv"
	@echo ""
	@echo "  Pipeline"
	@echo "    run          Pipeline: Default config (ORB + vocab 100 + knn)"
	@echo "    baseline     Pipeline: ORB + vocab 100 + knn        [yếu nhất]"
	@echo "    optimal      Pipeline: SIFT + vocab 1000 + svm_rbf  [mạnh nhất]"
	@echo ""
	@echo "  Clean"
	@echo "    clean-final  Xóa output/final/ và checkpoints"
	@echo "    clean        Xóa toàn bộ output/"
	@echo ""

install:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip -q
	.venv/bin/pip install -r requirements.txt -q
	@echo "Xong. Dùng: make run"

run:
	$(PYTHON) $(SCRIPT) $(ARGS)

baseline:
	$(PYTHON) $(SCRIPT) --feature orb --vocab-size 100 --classifier knn

optimal:
	$(PYTHON) $(SCRIPT) --feature sift --vocab-size 1000 --classifier svm_rbf

clean-final:
	rm -rf output/final output/model_*.pkl
	@echo "Đã xóa final + checkpoints"

clean:
	rm -rf output/
	@echo "Đã xóa toàn bộ output"
