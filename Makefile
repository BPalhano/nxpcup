.PHONY: install sync lint format typecheck test clean

install:
	@echo "==> Instalando dependências com uv..."
	uv pip install -e .

lint:
	@echo "==> Executando Ruff linter..."
	uv run ruff check main/src main/data main/tests

format:
	@echo "==> Formatando código com Ruff..."
	uv run ruff format main/src main/data main/tests

typecheck:
	@echo "==> Verificando tipos com Mypy..."
	uv run mypy --explicit-package-bases main/data main/src

test:
	@echo "==> Executando testes com Pytest..."
	uv run pytest main/tests -v

auto-label:
	@echo "==> Executando amostragem dos vídeos (10 FPS) e auto-labeling 3D..."
	uv run python main/data/auto_label_pipeline.py

train-microvit:
	@echo "==> Treinando MicroViT com MLflow..."
	uv run --directory main python -m src.train --model micro_vit

train-dave2:
	@echo "==> Treinando NVIDIA DAVE-2 com MLflow..."
	uv run --directory main python -m src.train --model dave2

train-palhano:
	@echo "==> Treinando PalhanoCNN com MLflow..."
	uv run --directory main python -m src.train --model palhano_cnn

clean:
	@echo "==> Limpando caches e arquivos temporários..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf *.egg-info
