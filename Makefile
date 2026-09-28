.PHONY: install corpus test eval robustness extraction cuad cuad-eval serve docker

install:
	pip install -r requirements.txt

corpus:
	python data/generate_corpus.py --n 200 --seed 7

test:
	python -m pytest tests/ -q

eval:
	python -m app.eval.run_eval --corpus data/corpus --out eval_report.json

robustness:
	python -m app.eval.paraphrase --corpus data/corpus --show holdout

extraction:
	python -m app.eval.extraction_eval --synthetic data/corpus

# Real contracts: 510 CUAD agreements with lawyers' clause labels (18 MB, public).
cuad:
	curl -L -o /tmp/cuad.zip https://github.com/TheAtticusProject/cuad/raw/main/data.zip
	unzip -o -q /tmp/cuad.zip -d /tmp/cuad
	python -m app.ingest.cuad --cuad /tmp/cuad/CUADv1.json --out data/cuad

cuad-eval:
	python -m app.eval.extraction_eval --cuad data/cuad

serve:
	uvicorn app.main:app --reload --port 8000

docker:
	docker build -t contract-intel . && docker run -p 8000:8000 contract-intel
