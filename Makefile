.PHONY: install sync lint format typecheck test clean auto-label train-microvit train-dave2 drift benchmark export-onnx api docker-build docker-run docker-compose-up

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

drift:
	@echo "==> Executando análise de Data Drift com Evidently AI..."
	uv run --directory main python -m src.monitoring.drift_report

benchmark:
	@echo "==> Executando benchmark estatístico de inferência..."
	uv run --directory main python -m src.inference.benchmark

export-onnx:
	@echo "==> Exportando modelos para ONNX otimizado..."
	uv run --directory main python -m src.optimization.onnx_exporter --model all

api:
	@echo "==> Iniciando serviço de inferência FastAPI com Uvicorn..."
	uv run uvicorn main.src.api.app:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	@echo "==> Construindo imagem Docker multi-stage (nxpcup-vision:latest)..."
	docker build -t nxpcup-vision:latest -f main/docker/Dockerfile .

docker-run:
	@echo "==> Executando container nxpcup-vision em background..."
	docker run -d -p 8000:8000 --name nxpcup-vision-api nxpcup-vision:latest

docker-compose-up:
	@echo "==> Subindo stack completa (FastAPI + MLflow) com Docker Compose..."
	docker compose -f main/docker/docker-compose.yml up -d

clean:
	@echo "==> Limpando caches e arquivos temporários..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf *.egg-info
	rm -f docs/*.aux docs/*.log docs/*.out docs/*.fls docs/*.fdb_latexmk docs/*.synctex.gz
