# Data

This directory contains one canonical copy of each dataset used in the paper.

- `main_prompts.csv` — 2,900 prompts across ten topic categories.
- `main_corpus.csv` — paired Gemma 3 and Qwen 2.5 responses to those prompts, giving 5,800 generated texts.
- `independent_test_prompts.csv` — 500 later prompts that are disjoint from the main prompt set.
- `independent_test_corpus.csv` — paired responses to the independent prompts, giving 1,000 generated test texts.
- `generation_protocol.json` — model identifiers, prompt wrapping, decoding parameters, serving software, and regeneration note.

The CSV files use semicolon delimiters. Prompt IDs preserve the identifiers used during the experiments.

The main prompt file contains 2,900 prompt records. Four prompt strings each occur twice, giving 2,896 distinct prompt strings; the released records exactly match the prompt multiset in `main_corpus.csv`.

All text is machine-generated. Email-like addresses and named entities appearing inside responses are part of the generated corpus and are not credentials or author contact data.

These data and their released compilation are licensed under [CC BY 4.0](../LICENSE-DATA). Please attribute the associated paper and indicate any modifications.
