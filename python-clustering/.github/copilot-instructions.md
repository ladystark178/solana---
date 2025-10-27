## Repo quick-start for AI coding agents

This project clusters and classifies Solana "meme" tokens using a pre-trained text model. Keep guidance short and actionable so an AI agent can be productive immediately.

- Big picture: `clustering.py` exposes a single domain model `MemeCoinClassifier` (loads a joblib package containing a `vectorizer`, `kmeans`, `cluster_descriptions`, and `metadata`) and a helper `cluster_tokens(tokens)` which returns a list of topics. Training/experimentation is in `meme_coin_classifierML.ipynb` and source data is `Meme_Coins_Solana_3000.csv`.

- Key files:
  - `clustering.py` — runtime inference + clustering helper. Important symbols: `MemeCoinClassifier`, `get_classifier()`, `cluster_tokens()`.
  - `meme_coin_classifierML.ipynb` — notebook used to build/preprocess data and produce the joblib model package.
  - `lightweight_model.json` — light artifact (model metadata or small model representation).
  - `Meme_Coins_Solana_3000.csv` — canonical dataset of tokens used for experiments.
  - `text_only_meme_classifier.pkl` — default model filename loaded by `MemeCoinClassifier` (may be produced by the notebook).

- Runtime patterns and contracts
  - Inputs: `cluster_tokens(tokens)` expects a list of dicts where each dict contains at least `name` and `symbol` keys (strings). Example token: `{ "name": "My Coin", "symbol": "MYC" }`.
  - Output: list of cluster dicts with keys: `topic_id`, `topic_name`, `keywords`, `tokens`, `similarity_threshold`, `confidence_score`, `cluster_info`.
  - Confidence semantics: `confidence_score` is computed from kmeans distances and sorted descending by the code. Handle missing cluster descriptions (the code uses fallback values and a `topic_fallback`).

- Language & preprocessing conventions
  - Text preprocessing handles both English and Chinese. Chinese tokenization uses `jieba` in `preprocess_text()`; English words are lower-cased and non-word characters removed.
  - Keyword extraction uses simple regex-based extraction (`extract_keywords`) and returns the most frequent words.

- Dependencies & testing
  - `requirements.txt` is currently empty — infer required packages from code: `joblib`, `numpy`, `scikit-learn`, `jieba` and `collections` (stdlib). When adding or updating dependencies, prefer pinning minimal versions. Example install: `pip install joblib numpy scikit-learn jieba`.
  - There is a small integration test helper sketch in the project checkpointed notebook that posts to `CLUSTER_SERVICE_URL` `/cluster`. Use the env var `CLUSTER_SERVICE_URL` when running remote integration tests.

- Common failure modes to detect and how to fix
  - Model load failure: joblib.load will raise; check the model file path (`text_only_meme_classifier.pkl`) and ensure the joblib package contains `vectorizer`, `kmeans`, `cluster_descriptions`, `metadata`.
  - Missing Chinese segmentation: ensure `jieba` is present and that `preprocess_text()` is used consistently before vectorization.
  - Unexpected output shape from vectorizer/kmeans: validate that `vectorizer.transform([text])` returns the expected sparse array and that `kmeans.predict`/`transform` are available.

- Guidance for edits
  - Small changes to clustering behavior should be made in `clustering.py`. Preserve the public function `cluster_tokens(tokens)` signature for backwards compatibility.
  - When adding new fields to cluster output, update any consumer code (notebooks, tests, API) that relies on keys like `topic_id`/`topic_name`/`confidence_score`.

- Example usage (Python):
  ```py
  from clustering import cluster_tokens
  tokens = [{"name":"DogCoin","symbol":"DOGE"}, {"name":"CatToken","symbol":"CAT"}]
  clusters = cluster_tokens(tokens)
  # clusters -> list of dicts described above
  ```

If anything here is unclear or you'd like the instructions to include CI/build commands or actual pinned dependency versions, tell me which area to expand and I will iterate.
