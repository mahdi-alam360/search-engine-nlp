# Document Search API

An AI-assisted learning prototype developed in October 2026. This is new work extending the ideas in the separately documented 2022 retrieval project; it is not a recovered implementation of that project.

## Run

Requires Python 3.11 or newer, with no third-party packages.

```sh
python -m unittest -v
python app.py
```

Open `http://127.0.0.1:8765/search?q=BM25&k=3`. For evidence responses, send a JSON POST to `/answer` with `{"query":"What is BM25?"}`. `/health` reports corpus size. All sample corpus notes are original explanatory text created for this demonstration.

## Capabilities

- BM25 ranking with document length normalization and deterministic ordering
- JSON REST endpoints for retrieval, health, and evidence responses
- Source identifiers included with extractive responses
- Query validation, request size limits, and API error responses
- Optional local Ollama generation adapter; set `OLLAMA_MODEL` to an installed model name before starting
- Docker recipe and HTTP integration tests

## Honest scope

The default response is retrieved evidence, not LLM-generated prose. The local generation adapter is implemented but has not been tested against a real running LLM in this environment. It uses the [Ollama generate API](https://docs.ollama.com/api/generate) at localhost. Model output is explicitly unverified; displaying citations does not prove a claim is grounded. This prototype has no agent orchestration, vector database, authentication, cloud deployment, or production-scale evaluation. The toy corpus and four relevance cases are functional checks, not an independent benchmark. The Python standard-library HTTP server is for local demonstrations, not internet-facing production use.

## Evaluation

Run `python evaluate.py` for the four hand-written relevance cases and a no-evidence case. Results describe only this tiny corpus. Test runtime and network/model performance depend on your environment.

Before discussing the project in an interview, run it, read the BM25 implementation, inspect its limitations, and be able to explain the difference between lexical retrieval and RAG with an actual language model.
