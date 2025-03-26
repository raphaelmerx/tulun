import re
import spacy
import time
from datasets import load_dataset
from typing import List, Tuple, Literal, Optional
from dataclasses import dataclass
from dotenv import load_dotenv 
load_dotenv()

from language_utils import languages, Language

nlp = spacy.load('en_core_web_sm')


@dataclass(frozen=True)
class CorpusEntry:
    src_lang: str
    tgt_lang: str
    src_text: str
    tgt_text: str

    @property
    def tokens(self) -> List[str]:
        text = re.sub(r'[^\w\s]', '', self.src_text)
        doc = nlp(text)
        return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]


class NLLBDevTest:
    def __init__(self):
        self.flores_dataset = load_dataset('openlanguagedata/flores_plus', split='devtest')
        self.lines = {}
        self.get_flores_lines('eng_Latn')
    
    def get_nllb_translation(self, en_text: str, lang_code: str) -> str:
        # relevant file example: flores_translations/flores200-eng_Latn-tpi_Latn-devtest.hyp
        with open(f'flores_translations/flores200-eng_Latn-{lang_code}-devtest.hyp') as f:
            tgt_lines = f.readlines()
            tgt_lines = [x.strip() for x in tgt_lines]
        en_to_tgt = dict(zip(self.lines["eng_Latn"], tgt_lines))
        try:
            return en_to_tgt[en_text]
        except KeyError:
            return ''
    
    def get_reference(self, en_text: str, lang_code: str) -> str:
        """ Get the reference translation for a given English text in a target language """
        # index of the English line
        line_index = self.lines["eng_Latn"].index(en_text)
        return self.get_flores_lines(lang_code)[line_index]
    
    def get_flores_lines(self, lang_code: str) -> List[str]:
        start = time.time()
        if lang_code in self.lines:
            return self.lines[lang_code]
        print(f'Flores: loading lines for {lang_code}', end=' ')
        iso_639_3 = lang_code.split('_')[0]
        iso_15924 = lang_code.split('_')[1]
        # features: ['id', 'iso_639_3', 'iso_15924', 'glottocode', 'text', 'url', 'domain', 'topic', 'has_image', 'has_hyperlink', 'last_updated'],
        lines = self.flores_dataset.filter(lambda x: x['iso_639_3'] == iso_639_3 and x['iso_15924'] == iso_15924)
        self.lines[lang_code] = lines['text']
        print(f'took {time.time() - start:.2f} seconds, loaded {len(lines)} lines')
        return self.lines[lang_code]


def get_train_lines(tgt_lang: Language, limit: Optional[int] = None) -> List[CorpusEntry]:
    print(f"Train lines: loading for {tgt_lang}", end=' ')
    tgt_lang = tgt_lang.nllb_code
    start = time.time()
    try:
        dataset = load_dataset("allenai/nllb", f"{tgt_lang}-eng_Latn", trust_remote_code=True)
    except ValueError:
        # try the other direction
        dataset = load_dataset("allenai/nllb", f"eng_Latn-{tgt_lang}", trust_remote_code=True)
    lines = dataset['train']
    # only keep lines with 5 words or more
    lines = lines.filter(lambda x: len(x['translation']['eng_Latn'].split()) > 5)
    if limit:
        try:
            lines = lines.select(range(limit))
        except IndexError:
            # happens when limit is larger than the number of lines
            pass
    if not lines:
        raise Exception(f"Language {tgt_lang} not found in NLLB")
    corpus_entries = [CorpusEntry(src_lang='eng_Latn', tgt_lang=tgt_lang, src_text=line['translation']['eng_Latn'], tgt_text=line['translation'][tgt_lang]) for line in lines]
    # dedup: unique by src_text
    corpus_entries = list({entry.src_text: entry for entry in corpus_entries}.values())
    print(f"took {time.time() - start:.2f} seconds, loaded {len(lines)} lines")
    return corpus_entries

def get_train_lines_smol(tgt_lang: Language, limit: Optional[int] = None) -> List[CorpusEntry]:
    tgt_lang_code = tgt_lang.openwho_code
    dataset = load_dataset("google/smol", f"smolsent__en_{tgt_lang_code}", trust_remote_code=True)
    lines = dataset['train']
    if not lines:
        raise Exception(f"Language {tgt_lang} not found in SmoL")
    corpus_entries = [CorpusEntry(src_lang='eng_Latn', tgt_lang=tgt_lang.nllb_code, src_text=line['src'], tgt_text=line['trg']) for line in lines]
    return corpus_entries