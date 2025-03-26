import os
import html
import json
import difflib
import litellm
import transformers
from functools import cached_property
from google.cloud import translate_v2 as translate
from typing import List, Literal, Optional, Tuple
from dataclasses import dataclass
import spacy
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()
import dspy

from .models import GlossaryEntry, CorpusEntry, SystemConfiguration
from .dspy_models import Input, PostEditSignature

nlp = spacy.load('en_core_web_sm')

config = SystemConfiguration.load()

@dataclass(frozen=True)
class Message:
    role: Literal['user', 'system', 'assistant']
    content: str

    @staticmethod
    def format_query(line: CorpusEntry, google_translated: str) -> 'Message':
        return Message(role='user', content=f"<English>{line.english_text}\n<{config.target_language_name} MT>{google_translated}</{config.target_language_name} MT>")
    
    @staticmethod
    def format_response(line: CorpusEntry) -> 'Message':
        return Message(role='assistant', content=f"<{config.target_language_name} (post-edited)>{line.translated_text}</{config.target_language_name} (post-edited)>")
    
    def as_dict(self):
        return {
            'role': self.role,
            'content': self.content,
        }



class TranslatorMixin:
    def __init__(self) -> None:
        # access it now, outside of an async loop
        self.top_n = config.num_sentences_retrieved
        self.config = config

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

    async def construct_prompt_post_edit(self, sent: str, sent_mt: str, top_similar_sentences: List[CorpusEntry], glossary_entries: List[GlossaryEntry]) -> List[Message]:
        messages = [
            Message(role='system', content=config.translation_prompt),
        ]

        user_message = ''

        if glossary_entries:
            user_message += "<glossary entries>\n"
            for i, entry in enumerate(glossary_entries):
                user_message += f"no {i}: {entry.as_txt()}\n"
            user_message += "</glossary entries>\n\n"

        user_message += "<past translations>\n"
        for i, sentence in enumerate(top_similar_sentences):
            sentence_mt = self.translate(sentence.english_text)
            user_message += f"English: {sentence.english_text}\nMachine translated: {sentence_mt}\n{config.target_language_name}: {sentence.translated_text}\n\n"
        user_message += "</past translations>\n\n"
        
        user_message += "Text to translate:\n"

        user_message += f"English: {sent}\nMachine translated: {sent_mt}\n{config.target_language_name}: "

        messages.append(Message(role='user', content=user_message))
        # for m in messages:
        #     print(f"{m.role}: {m.content}")
        return messages

    async def get_post_edited_translation(self, input_text: str, similar_sentences, glossary_entries) -> str:
        if config.dspy_config:
            return await self.get_post_edited_translation_dspy(input_text, similar_sentences, glossary_entries)

        sent_mt = self.translate(input_text)
        messages = await self.construct_prompt_post_edit(
            input_text, 
            sent_mt,
            similar_sentences,
            glossary_entries=glossary_entries,
        )

        response = litellm.completion(
            model=config.post_editing_model,
            messages=[m.as_dict() for m in messages],
            temperature=0.5,
        )

        response = response.choices[0].message.content
        final_translation = response.strip()
        return {
            "final_translation": final_translation,
            "corrections": Correction.from_string_matching(sent_mt, final_translation),
        }

    async def get_post_edited_translation_dspy(self, input_text: str, similar_sentences: List[CorpusEntry], glossary_entries: List[GlossaryEntry]) -> str:

        predictor = dspy.Predict(PostEditSignature)
        predictor.load_state(json.loads(config.dspy_config))

        machine_translated = self.translate(input_text)

        input = Input(
            input_text=input_text,
            machine_translated=machine_translated,
            glossary_entries=[ge.as_dict() for ge in glossary_entries],
            past_translations=[line.as_dict() for line in similar_sentences]
        )

        prediction = predictor(input=input)
        final_translation = prediction.output.output_text
        return {
            "final_translation": final_translation,
            "corrections": Correction.from_string_matching(machine_translated, final_translation),
        }


class TranslatorGoogle(TranslatorMixin):
    def __init__(self) -> None:
        self.memory_filename = 'datafiles/google_translations.json'
        self.gclient = translate.Client()
        self.load_translation_memory()
        super().__init__()
    
    def translate(self, text: str) -> str:
        if text in self._translation_memory:
            return self._translation_memory[text]
        translation = self.gclient.translate(text, source_language='en', target_language=config.target_language_code)
        translation = translation['translatedText']
        self._translation_memory[text] = translation
        self.save_translation_memory()
        return translation


class TranslatorHuggingFace(TranslatorMixin):
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-tdt"):
        self.translator = transformers.pipeline("translation", model=model_name)
        model_name_for_filename = model_name.replace("/", "_")
        self.memory_filename = f'datafiles/huggingface_{model_name_for_filename}.json'
        self.load_translation_memory()
        super().__init__()
    
    def translate(self, text: str) -> str:
        if text in self._translation_memory:
            return self._translation_memory[text]
        translation = self.translator(text)[0]['translation_text']
        self._translation_memory[text] = translation
        self.save_translation_memory()
        return translation


@dataclass
class Correction:
    from_: str
    to: str

    def as_dict(self):
        return {
            'from': self.from_,
            'to': self.to,
        }

    @classmethod
    def from_string_matching(cls, s1, s2) -> List['Correction']:
        """
        Identifies edits made to transform s1 into s2.
        Returns a list of dictionaries with 'from' and 'to' keys.
        """
        s1 = html.unescape(s1)
        s2 = html.unescape(s2)
        # s1, s2: ignore trailing punctuation
        s1 = s1.strip('.!? \n').replace("’", "'")
        s2 = s2.strip('.!? \n').replace("’", "'")
        words1 = s1.split()
        words2 = s2.split()
        
        matcher = difflib.SequenceMatcher(None, words1, words2)
        opcodes = matcher.get_opcodes()
        
        corrections = []
        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'replace':
                corrections.append(cls(
                    from_=' '.join(words1[i1:i2]),
                    to=' '.join(words2[j1:j2])
                ))
            elif tag == 'delete':
                corrections.append(cls(
                    from_=' '.join(words1[i1:i2]),
                    to=' '.join(words2[j1:j2])
                ))
            elif tag == 'insert':
                corrections.append(cls(
                    from_=' '.join(words1[i1:i2]),
                    to=' '.join(words2[j1:j2])
                ))
        
        # Filter out empty edits
        corrections = [c for c in corrections if c.from_ != c.to]
        
        return corrections

    @classmethod
    def from_llm_response(cls, response: str, input_text: str, final_translation: str) -> List['Correction']:
        try:
            rationale = response.split("<rationale>")[1].split("</rationale>")[0]
        except IndexError:
            return []
        rationale = [l.strip() for l in rationale.split('\n') if l.strip()]
        corrections = []
        # get the <English> part, <machine translated> part, and <corrected> part for each line
        for line in rationale:
            try:
                english = line.split("<English>")[1].split("</English>")[0]
                if english not in input_text:
                    continue
                mt = line.split("<machine translated>")[1].split("</machine translated>")[0]
                corrected = line.split("<corrected>")[1].split("</corrected>")[0]
                corrections.append(cls(from_=mt, to=corrected))
            except IndexError:
                continue
        # sanity check:
        # - remove if mt and corrected are the same
        corrections = [c for c in corrections if c.from_ != c.to]
        # - remove if corrected is not in the translation
        corrections = [c for c in corrections if c.to in final_translation]
        return corrections