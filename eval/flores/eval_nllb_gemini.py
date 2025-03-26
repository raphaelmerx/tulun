import sys
import os
import csv
import random
import json
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataclasses import dataclass
from typing import List, Optional, Literal, Tuple
from dotenv import load_dotenv
load_dotenv()

from tqdm import tqdm
from sacrebleu import corpus_chrf, corpus_bleu

from language_utils import languages, Language
from flores_utils import NLLBDevTest, get_train_lines, CorpusEntry, get_train_lines_smol
from gatitos_utils import Glossary
from translator import Translator


parser = argparse.ArgumentParser(description="Eval NLLB+LLM for low-resource")
parser.add_argument("--tgt-lang", type=str, required=True, help="Target language, e.g. 'tpi_Latn")
parser.add_argument("--limit", type=int, default=None, help="Number of test lines to evaluate")
parser.add_argument('--train-set', type=str, default='nllb', help="Train set to use for retrieval")
parser.add_argument("--no-output", action="store_true", help="Don't print results to eval_results.jsonl", default=False)
parser.add_argument("--debug", action="store_true", help="Print debug info", default=False)
parser.add_argument("--no-glossary", action="store_true", help="Don't use glossary", default=False)
parser.add_argument("--translate-only", action="store_true", help="Only translate using an LLM, no post-editing", default=False)
parser.add_argument("--translate-zero-shot", action="store_true", help="Translate zero-shot using an LLM", default=False)
args = parser.parse_args()

language: Language = languages.get_by_nllb_code(args.tgt_lang)
if not language:
    raise Exception(f"Language {args.tgt_lang} not found")

nllb = NLLBDevTest()
if args.train_set == 'nllb':
    train_lines = get_train_lines(language, limit=1_000_000)
elif args.train_set == 'smol':
    train_lines = get_train_lines_smol(language, limit=None)

test_lines = nllb.get_flores_lines("eng_Latn").copy()
random.seed(42)
random.shuffle(test_lines)
if args.limit:
    test_lines = test_lines[:args.limit]

glossary = Glossary()
glossary.load(language.openwho_code)

translator = Translator(tgt_lang = language, nllb = nllb, debug=args.debug, use_glossary=not args.no_glossary)
translator.init_bm25(train_lines)

def get_random_train_lines(num: int = 10) -> List[Tuple[CorpusEntry, float]]:
    lines = random.sample(train_lines, num)
    return [(line, 0.0) for line in lines]

results = []

for line in tqdm(test_lines):
    src_text = line
    glossary_entries = glossary.get_entries(src_text)
    similar_sentences = translator.get_top_similar_sentences_bm25(src_text)
    # similar_sentences = get_random_train_lines(10)  # used for fixed_gemini
    if args.translate_only:
        final_translation = translator.get_translation(src_text, similar_sentences)
    elif args.translate_zero_shot:
        final_translation = translator.get_zero_shot_translation(src_text)
    else:
        try:
            final_translation = translator.get_post_edited_translation(src_text, similar_sentences, glossary_entries)
        except Exception as e:
            print(f"Error: {e}")
            continue
    reference = nllb.get_reference(src_text, language.nllb_code)
    mt = nllb.get_nllb_translation(src_text, language.nllb_code)
    results.append({'hyp': final_translation, 'mt': mt, 'ref': reference})

    if args.debug:
        print(f"Source: {src_text}")
        print(f"MT: {mt}")
        print(f"post edited: {final_translation}")
        print(f"Reference: {reference}")

mt_chrf = corpus_chrf(
    [r['mt'] for r in results],
    [[r['ref'] for r in results]],
    word_order=2
)
print(f"CHRF for MT: {mt_chrf.score:.2f}")

mt_post_edit_chrf = corpus_chrf(
    [r['hyp'] for r in results],
    [[r['ref'] for r in results]],
    word_order=2
)
print(f"CHRF for MT + post-editing: {mt_post_edit_chrf.score:.2f}")
print(f"Lang: {language.name}, limit: {args.limit}")

if not args.no_output and not args.debug:
    with open('eval_results.jsonl', 'a') as f:
        res = {
            "tgt_lang": language.nllb_code,
            "chrf_mt": float(f"{mt_chrf.score:.2f}"),
            "chrf_mt_ape": float(f"{mt_post_edit_chrf.score:.2f}"),
            "diff": float(f"{mt_post_edit_chrf.score - mt_chrf.score:.2f}"),
            "glossary": not args.no_glossary,
            "train": args.train_set,
            "limit": args.limit or len(results)
        }
        if args.translate_only:
            res['translate_only'] = True
        elif args.translate_zero_shot:
            res['translate_zero_shot'] = True
        f.write(json.dumps(res) + '\n')