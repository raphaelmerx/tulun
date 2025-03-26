import re
import os
import time
from google.cloud import translate_v2 as translate
from typing import List, Literal, Optional
import spacy
import pandas as pd
import json
import tantivy
import logging
import csv
from tqdm import tqdm

from rank_bm25 import BM25Okapi
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass

nlp = spacy.load('en_core_web_sm')

@dataclass
class Config:
    src_lang_name: str = 'English'
    tgt_lang_name: str = 'Tetun'
    src_lang_code: str = 'en'
    tgt_lang_code: str = 'tdt'

config = Config(src_lang_name='English', tgt_lang_name='Tetun', src_lang_code='en', tgt_lang_code='tdt')
# config = Config(src_lang_name='English', tgt_lang_name='Bislama', src_lang_code='en', tgt_lang_code='bis')


@dataclass
class Line:
    en: str
    tgt: str
    tgt_pred_gt: str = None
    tgt_pred_madlad: str = None
    tgt_pred_opusmt: str = None
    split: Optional[Literal['train', 'test']] = None

    @property
    def tokens(self):
        text = re.sub(r'[^\w\s]', '', self.en)
        doc = nlp(text)
        return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    
    def as_txt(self):
        return f"{config.src_lang_name}: {self.en}\n{config.tgt_lang_name}: {self.tgt}"

class LineManager:
    def __init__(self, lines: List[Line]) -> None:
        self.lines = lines
        self.split_train_test()
    
    def split_train_test(self):
        for i, line in enumerate(self.lines):
            line.split = 'train' if i < 0.5 * len(self.lines) else 'test'
    
    @classmethod
    def load_from_csv(cls, filename: str = 'datafiles/tetun_parallel.csv') -> 'LineManager':
        with open(filename) as f:
            reader = csv.DictReader(f)
            lines = [Line(**row) for row in reader]
        return cls(lines)

    @property
    def train_lines(self) -> List[Line]:
        return [l for l in self.lines if l.split == 'train']
    
    @property
    def test_lines(self) -> List[Line]:
        return [l for l in self.lines if l.split == 'test']
    

@dataclass
class GlossaryEntry:
    en: str
    tgt: str

    def as_txt(self):
        return f"{self.en} -> {self.tgt}"

    def as_dict(self) -> dict:
        return {
            'key': self.en,
            'value': self.tgt,
        }

@dataclass
class Message:
    role: Literal['user', 'system', 'assistant']
    content: str

    @staticmethod
    def format_query(line: Line, google_translated: str) -> 'Message':
        return Message(role='user', content=f"English: {line.en}\n{config.tgt_lang_name} (from Google Translate): {google_translated}\{config.tgt_lang_name}:")
    
    @staticmethod
    def format_response(line: Line) -> 'Message':
        return Message(role='assistant', content=f"{line.tgt}")
    
    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content,
        }
    
    def as_txt(self):
        if self.role == 'system':
            return f"<|begin_of_text|><|start_header_id|>{self.role}<|end_header_id|>\n{self.content}<|eot_id|>"
        return f"<|start_header_id|>{self.role}<|end_header_id|>\n{self.content}\n"


class MessageList(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    

@dataclass
class Glossary:
    entries: List[GlossaryEntry] = None
    file: str = 'datafiles/glossary_medical_eng_tdt.json'

    def get_entry(self, en: str) -> Optional[GlossaryEntry]:
        try:
            return next(entry for entry in self.entries if entry.en == en)
        except StopIteration:
            return None
    
    def load_entries(self):
        with open(self.file) as f:
            entries = json.load(f)
            entries = [e for e in entries if e['key'] and e['value']]
        self.entries = []
        for entry in tqdm(entries, desc="Loading glossary entries"):
            # tokenize the key
            key = entry['key']
            key = ' '.join([token.lemma_.lower() for token in nlp(key)])
            self.entries.append(GlossaryEntry(en=key, tgt=entry['value']))
    
    def save_entries(self):
        with open(self.file, 'w') as f:
            json.dump([entry.as_dict() for entry in self.entries], f, indent=2)

    def get_entries(self, sentence: str) -> List[GlossaryEntry]:
        doc = nlp(sentence)
        tokens = [token.lemma_.lower() for token in doc]
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        entries = [self.get_entry(key) for key in tokens + bigrams]
        entries = [entry for entry in entries if entry]
        # unique by en
        entries = list({entry.en: entry for entry in entries}.values())
        return entries
    
    def save(self, df: pd.DataFrame):
        entries = df.to_dict(orient='records')
        for df_entry in entries:
            key = df_entry['English']
            entry = self.get_entry(key)
            if entry:
                entry.tgt = df_entry['Tetun']
            else:
                self.entries.append(GlossaryEntry(en=key, tgt=df_entry['Tetun']))
        self.save_entries()


class Translator:
    def __init__(self, translate_with: Literal["google", "madlad", "opusmt"] = "google") -> None:
        if translate_with == "google":
            self.gclient = translate.Client()
            self.load_google_memory()
        elif translate_with.lower() == "madlad":
            self.load_madlad_memory()
        elif translate_with == "opusmt":
            self.load_opusmt_memory()
        self.translate_with = translate_with

    def init_bm25(self, lines: List[Line]) -> BM25Okapi:
        start = time.time()
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("text", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("tetun_text", stored=True)
        schema_builder.add_integer_field("id", stored=True)  # Add ID field
        schema = schema_builder.build()

        self.index = tantivy.Index(schema)

        writer = self.index.writer()
        for i, line in enumerate(lines):
            writer.add_document(tantivy.Document(
                text=line.en,
                tgt_text=line.tgt,
                id=i  # Store the index
            ))
        writer.commit()
        writer.wait_merging_threads()

        self.lines_bm25 = lines
        end = time.time()
        print("Translator: Tantivy index initialized, took {:.2f} seconds".format(end - start))

    def load_madlad_memory(self):
        if config.tgt_lang_code == 'tdt':
            filename = "datafiles/tetun_parallel.csv"
        elif config.tgt_lang_code == 'bis':
            filename = "datafiles/bislama_parallel.csv"
        with open(filename) as f:
            reader = csv.DictReader(f)
            self._trans_memory = {row['en']: row['tgt_pred_madlad'] for row in reader}
    
    def load_opusmt_memory(self):
        if config.tgt_lang_code == 'tdt':
            filename = "datafiles/tetun_parallel.csv"
        elif config.tgt_lang_code == 'bis':
            filename = 'datafiles/bislama_parallel_gemini.csv'
        with open(filename) as f:
            reader = csv.DictReader(f)
            self._trans_memory = {row['en']: row['tgt_pred_opusmt'] for row in reader}
    
    def load_google_memory(self, filename: str = 'google_translation_memory.csv'):
        if not os.path.exists(filename):
            self._translation_memory = {}
            return
        with open(filename) as f:
            reader = csv.DictReader(f)
            self._translation_memory = {row['en']: row['tgt'] for row in reader}
    
    def save_translation_memory(self, filename: str = 'google_translation_memory.csv'):
        with open(filename, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['en', 'tgt'])
            writer.writeheader()
            writer.writerows([{'en': k, 'tgt': v} for k, v in self._translation_memory.items()])

    def translate(self, text: str) -> str:
        if self.translate_with == "google":
            return self.translate_google(text)
        elif self.translate_with.lower() == "madlad" or self.translate_with == "opusmt":
            return self.translate_mem_only(text)
        else:
            raise ValueError(f"Invalid translation method: {self.translate_with}")
        
    def translate_google(self, text: str) -> str:
        if text in self._translation_memory:
            return self._translation_memory[text]
        target_language = config.tgt_lang_code if config.tgt_lang_code != 'tdt' else 'tet'
        translation = self.gclient.translate(text, source_language='en', target_language=target_language)
        translation = translation['translatedText']
        self._translation_memory[text] = translation
        self.save_translation_memory()
        return translation
    
    def translate_mem_only(self, text: str) -> str:
        return self._trans_memory[text]

    def get_top_similar_sentences_bm25(self, sent: str, top_n: int = 10) -> List[tuple[Line, float]]:
        self.index.reload()
        searcher = self.index.searcher()
        query_tokens = Line(en=sent, tgt='').tokens
        query = self.index.parse_query(' '.join(query_tokens), ["text"])
        search_results = searcher.search(query, top_n).hits

        results = []
        for score, doc_address in search_results:
            doc = searcher.doc(doc_address)
            # Use the ID to get the correct CorpusEntry
            doc_id = doc["id"][0]
            results.append((self.lines_bm25[doc_id], score))

        return results

    def construct_prompt_translation(self, sent: str, top_similar_sentences: List[Line], system_instruction: Optional[str] = None) -> List[Message]:
        if system_instruction:
            messages = [Message(role='system', content=system_instruction)]
        user_message = ''

        if top_similar_sentences:
            for sentence in top_similar_sentences:
                user_message += f"English: {sentence.en}\n{config.tgt_lang_name}: {sentence.tgt}\n\n"
        
        user_message += f"English: {sent}\n{config.tgt_lang_name}: "

        messages.append(Message(role='user', content=user_message))
        return messages

    def construct_prompt_post_edit(self, sent: str, top_similar_sentences: List[tuple[Line, float]], glossary_entries: List[GlossaryEntry]) -> List[Message]:
        user_message = ''

        if glossary_entries:
            user_message += "<glossary entries>\n"
            for i, entry in enumerate(glossary_entries):
                user_message += f"no {i}: {entry.as_txt()}\n"
            user_message += "</glossary entries>\n\n"
        
        user_message += "<past translations>\n"
        top_similar_sentences = [sentence for sentence, _ in top_similar_sentences]
        for i, sentence in enumerate(top_similar_sentences):
            sentence_mt = self.translate(sentence.en)
            user_message += f"English: {sentence.en}\nMT: {sentence_mt}\n{config.tgt_lang_name}: {sentence.tgt}\n\n"
        user_message += "</past translations>\n\n"
        
        user_message += "Text to translate:\n"

        sent_mt = self.translate(sent)
        user_message += f"English: {sent}\nMT: {sent_mt}\n{config.tgt_lang_name}: "

        messages = [Message(role='user', content=user_message)]
        return messages
