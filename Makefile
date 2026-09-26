.PHONY: install corpus test eval serve docker

install:
	pip install -r requirements.txt

corpus:
	python data/generate_corpus.py --n 200 --seed 7

test:
	python -m pytest tests/ -q

eval:
	python -m app.eval.run_eval --corpus data/corpus --out eval_report.json

serve:
	uvicorn app.main:app --reload --port 8000

docker:
	docker build -t contract-intel . && docker run -p 8000:8000 contract-intel
