import os
import json
from typing import List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass

from dotenv import load_dotenv
load_dotenv()

import dspy

from .models import GlossaryEntry, SystemConfiguration, CorpusEntry

lm = dspy.LM('gemini/gemini-2.0-flash', api_key=os.getenv('GEMINI_API_KEY'))
dspy.configure(lm=lm)

config = SystemConfiguration.load()

class GlossaryEntrySimplified(BaseModel):
    en: str = Field(title="The English key")
    tgt: str = Field(title="The translated entry")

class CorpusEntrySimplified(BaseModel):
    en: str = Field(title="The English text")
    tgt: str = Field(title="The translated text")
    machine_translation: str = Field(title="The machine translation of the English text", default="")


class Input(BaseModel):
    input_text: str = Field(title="The text to translate to Bislama")
    machine_translated: str = Field(title="A first-pass machine translation of the text")
    glossary_entries: List[GlossaryEntrySimplified] = Field(default_factory=list, description="A list of glossary entries to use for translation")
    past_translations: List[CorpusEntrySimplified] = Field(default_factory=list, description="A list of past translations to use for translation")

    @classmethod
    async def from_english_text(cls, english_text: str, translator):
        glossary_entries = GlossaryEntry.get_entries(english_text)
        glossary_entries = [ge.as_dict() for ge in glossary_entries]
        machine_translated = await translator.translate(english_text)
        similar_sentences = CorpusEntry.get_top_similar_bm25(english_text, config.num_sentences_retrieved)
        similar_sentences = [l for (l, _) in similar_sentences]
        similar_sentences = [l.as_dict() for l in similar_sentences]
        return cls(input_text=english_text, machine_translated=machine_translated, glossary_entries=glossary_entries, past_translations=similar_sentences)

class Output(BaseModel):
    output_text: str = Field(title="The translated text")

class PostEditSignature(dspy.Signature):
    input: Input = dspy.InputField()
    output: Output = dspy.OutputField()