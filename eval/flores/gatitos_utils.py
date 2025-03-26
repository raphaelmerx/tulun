import csv
import os
import json
from typing import List, Tuple, Literal
from dataclasses import dataclass, field
import spacy
from dotenv import load_dotenv
load_dotenv()
from typing import Optional
import datasets

from language_utils import languages

nlp = spacy.load('en_core_web_sm')
this_dir = os.path.dirname(os.path.abspath(__file__))
gatitos_dir = os.path.join(this_dir, 'url-nlp', 'gatitos')
    

@dataclass
class GlossaryEntry:
    src: str
    tgt: str

    def as_txt(self):
        return f"{self.src} -> {self.tgt}"

@dataclass
class Glossary:
    entries: List[GlossaryEntry] = field(default_factory=list)
    
    def get_entries(self, sentence: str) -> List[GlossaryEntry]:
        doc = nlp(sentence)
        tokens = [token.lemma_.lower() for token in doc]
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        entries = [e for e in self.entries if e.src in tokens or e.src in bigrams]
        return list(entries)
    
    def load(self, tgt_langcode: str):
        # new: load from google/smol
        ds = datasets.load_dataset('google/smol', f'gatitos__en_{tgt_langcode}')['train']
        for row in ds:
            for tgt in row['trgs']:
                self.entries.append(GlossaryEntry(row['src'], tgt))
        print(f"Gatitos: loaded {len(self.entries)} entries for {tgt_langcode}")

    def load_url_nlp(self, tgt_langcode: str):
        print(f"Gatitos: loading glossary for {tgt_langcode}", end=' ')
        if self.entries:
            raise Exception("Glossary already loaded")
        filename = os.path.join(gatitos_dir, f'en_{tgt_langcode}.tsv')
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                self.entries.append(GlossaryEntry(row[0], row[1]))
        print(f"found {len(self.entries)} entries")