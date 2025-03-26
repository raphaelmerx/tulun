# Tulun evaluations

First, install additional requirements:
```bash
pip install sacrebleu google-generativeai rank-bm25
```

## Tetun medical translation, Bislama disaster relief translations

Start with `cd tetun_bislama`, then:

#### 1. Get data files

To be placed in the [`datafiles`](./tetun_bislama/datafiles/) directory:
- Parallel lines (CSV with columns `en,tgt,tgt_pred_madlad,tgt_pred_opusmt`)
  - Tetun medical (903 total): `tetun_parallel.csv` 
  - Bislama lines (4417 total): `bislama_parallel.csv`
- Dictionaries (JSON array, where each entry has a `key` and `value` field)
  - Tetun medical dictionary: `glossary_medical_eng_tdt.json`
  - Bislama dictionary: `bislama_school_dictionary.json`

#### 2. Setup language config in [`translator_utils.py`](./tetun_bislama/translator_utils.py)

#### 3. Run evaluation scripts

- For Tetun: notebook [`tetun.ipynb`](./tetun_bislama/tetun.ipynb)
- For Bislama: notebook [`bislama.ipynb`](./tetun_bislama/bislama.ipynb)


## FLORES-200

Start with `cd flores`, then:
- for NLLB + Gemini post-editing: `python eval_nllb_gemini.py --tgt-lang <nllb-lang-code>`. Will output results (MT baseline and post-edited) in `eval_results.jsonl`
- for Gemini 10-shot: `python eval_nllb_gemini.py --tgt-lang <nllb-lang-code> --translate-only`