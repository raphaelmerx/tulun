import csv
import os
import time
import json
from typing import List, Tuple, Literal
from dataclasses import dataclass
from dotenv import load_dotenv 
load_dotenv()
from typing import Optional
import numpy as np
from rank_bm25 import BM25Okapi
import tantivy


import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

from language_utils import languages, Language
from gatitos_utils import GlossaryEntry
from flores_utils import NLLBDevTest, CorpusEntry
from gemini_client import GenerativeModel, Message



class Translator:
    def __init__(self, tgt_lang: Language, nllb: NLLBDevTest, debug: bool = False, use_glossary: bool = True):
        self.tgt_lang = tgt_lang
        self.nllb = nllb
        self.use_glossary = use_glossary
        self.client = GenerativeModel(system_instruction=self.system_prompt, generation_config={'temperature': 0.5})
        self.debug = debug
    
    def translate(self, text: str) -> str:
        return self.nllb.get_nllb_translation(text, self.tgt_lang.nllb_code)

    @property
    def system_prompt(self):
        return f"""You are a linguist helping to post-edit translations from English to {self.tgt_lang.name}

You will be provided with:
{"- a list of glossary entries that are relevant to the text to translate" if self.use_glossary else ""}
- a list of past translations of similar sentences
- a machine translation of the sentence to translate

Your task is to correct the machine translation using the {"glossary entries and " if self.use_glossary else ""}past translations.
Only use the {"glossary entries and " if self.use_glossary else ""}past translations to inform your edits, and nothing else.
If some {"glossary entries or " if self.use_glossary else ""}past translations are not relevant to this context, ignore them.

In your answer, wrap the corrected final final translation in the following tags: <corrected {self.tgt_lang.name}>...</corrected {self.tgt_lang.name}>"""

    @property
    def system_prompt_translation(self):
        # from SMOL paper
        return f"""You are an expert translator. I am going to give you some example pairs of text snippets where the first is in English and the second is a translation of the first snippet into {self.tgt_lang.name}. The sentences will be written English: <first sentence> {self.tgt_lang.name}: <translated first sentence> After the example pairs, I am going to provide another sentence in English and I want you to translate it into {self.tgt_lang.name}. Give only the translation, and no extra commentary, formatting, or chattiness. Translate the text from English to {self.tgt_lang.name}."""
    
    @property
    def system_prompt_translation_zero_shot(self):
        # adapted from SMOL paper
        return f"You are an expert translator. I am going to give you text in English, and would like you to translate it to {self.tgt_lang.name}. Give only the translation, and no extra commentary, formatting, or chattiness."

    def load_translation_memory(self):
        filename = self.memory_filename
        if not os.path.exists(filename):
            self._translation_memory = {}
            return
        with open(filename, 'r') as f:
            self._translation_memory = json.load(f)
    
    def save_translation_memory(self):
        with open(self.memory_filename, 'w') as f:
            json.dump(self._translation_memory, f, indent=2)

    def init_bm25(self, lines: List[CorpusEntry]) -> None:
        print("Translator: setting up BM25 index")
        start = time.time()
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("text", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("src_lang", stored=True)
        schema_builder.add_text_field("tgt_lang", stored=True)
        schema_builder.add_text_field("tgt_text", stored=True)
        schema_builder.add_integer_field("id", stored=True)  # Add ID field
        schema = schema_builder.build()

        self.index = tantivy.Index(schema)
        
        writer = self.index.writer()
        for i, line in enumerate(lines):
            writer.add_document(tantivy.Document(
                text=line.src_text,
                src_lang=line.src_lang,
                tgt_lang=line.tgt_lang,
                tgt_text=line.tgt_text if line.tgt_text else "",
                id=i  # Store the index
            ))
        writer.commit()
        writer.wait_merging_threads()
        
        self.lines_bm25 = lines
        end = time.time()
        print("Translator: Tantivy index initialized, took {:.2f} seconds".format(end - start))

    def get_top_similar_sentences_bm25(self, sent: str, top_n = 10) -> List[Tuple[CorpusEntry, float]]:
        self.index.reload()
        searcher = self.index.searcher()
        sent_tokens = CorpusEntry(src_lang='eng_Latn', tgt_lang=self.tgt_lang.nllb_code, src_text=sent, tgt_text='').tokens
        query = self.index.parse_query(' '.join(sent_tokens), ["text"])
        search_results = searcher.search(query, top_n).hits
        
        results = []
        for score, doc_address in search_results:
            doc = searcher.doc(doc_address)
            # Use the ID to get the correct CorpusEntry
            doc_id = doc["id"][0]
            results.append((self.lines_bm25[doc_id], score))
            
        return results

    def construct_prompt_translation(self, sent: str, top_similar_sentences: List[tuple[CorpusEntry, float]]) -> List[Message]:
        messages = [
            Message(
                role='system',
                content=self.system_prompt_translation,
            )
        ]
        user_message = ''

        top_similar_sentences = [sentence for sentence, _ in top_similar_sentences]
        for i, sentence in enumerate(top_similar_sentences):
            user_message += f"English: {sentence.src_text}\n{self.tgt_lang.name}: {sentence.tgt_text}\n\n"
        
        user_message += f"English: {sent}\n{self.tgt_lang.name}: "

        messages.append(Message(role='user', content=user_message))
        if self.debug:
            for m in messages:
                print(f"{m.role}: {m.content}")
        return messages

    def construct_prompt_post_edit(self, sent: str, top_similar_sentences: List[tuple[CorpusEntry, float]], glossary_entries: List[GlossaryEntry]) -> List[Message]:
        messages = [
            Message(
                role='system',
                content=self.system_prompt,
            )
        ]
        user_message = ''

        if glossary_entries and self.use_glossary:
            user_message += "<glossary entries>\n"
            for i, entry in enumerate(glossary_entries):
                user_message += f"no {i}: {entry.as_txt()}\n"
            user_message += "</glossary entries>\n\n"
        
        user_message += "<past translations>\n"
        top_similar_sentences = [sentence for sentence, _ in top_similar_sentences]
        for i, sentence in enumerate(top_similar_sentences):
            user_message += f"no {i}: <English>{sentence.src_text}</English><{self.tgt_lang.name}>{sentence.tgt_text}</{self.tgt_lang.name}>\n"
        user_message += "</past translations>\n\n"
        
        user_message += "Text to translate:\n"

        sent_gt = self.translate(sent)
        user_message += f"<English>{sent}</English>\n<machine translated>{sent_gt}</machine translated>"

        messages.append(Message(role='user', content=user_message))
        if self.debug:
            for m in messages:
                print(f"{m.role}: {m.content}")
        return messages
    
    def get_translation(self, input_text: str, similar_sentences: List[Tuple[CorpusEntry, float]]) -> str:
        messages = self.construct_prompt_translation(
            input_text, 
            similar_sentences,
        )
        messages = tuple(messages)
        client = GenerativeModel(system_instruction=self.system_prompt_translation, generation_config={'temperature': 0.5})
        response = client.get_response(messages)
        return response.strip()
    
    def get_zero_shot_translation(self, input_text: str) -> str:
        messages = [
            Message(
                role='user',
                content=f"English: {input_text}\n{self.tgt_lang.name}: "
            )
        ]
        if self.debug:
            print(messages[0].content)
        messages = tuple(messages)
        client = GenerativeModel(system_instruction=self.system_prompt_translation_zero_shot, generation_config={'temperature': 0.5})
        response = client.get_response(messages)
        return response.strip()

    def get_post_edited_translation(self, input_text: str, similar_sentences: List[Tuple[CorpusEntry, float]], glossary_entries: List[GlossaryEntry]) -> str:
        messages = self.construct_prompt_post_edit(
            input_text, 
            similar_sentences,
            glossary_entries=glossary_entries,
        )
        messages = tuple(messages)
        response = self.client.get_response(messages)
        try:
            final_translation = response.split(f"<corrected {self.tgt_lang.name}>")[1].split(f"</corrected {self.tgt_lang.name}>")[0]
            return final_translation.strip()
        except IndexError:
            print(f"Error with final translation: {response}")
            raise
