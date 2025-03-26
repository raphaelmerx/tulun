import re
import time
import tantivy
from typing import List, Optional
import spacy
from asgiref.sync import sync_to_async

from django.db import models
from django.db.utils import OperationalError
from django.core.exceptions import ValidationError
from django.contrib.auth.models import AbstractUser
from django.contrib.auth import get_user_model
from django.utils.functional import LazyObject

nlp = spacy.load('en_core_web_sm')


class CustomUser(AbstractUser):
    title = models.TextField(max_length=100, blank=True)
    email = models.EmailField('Email Address', unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

class GlossaryEntry(models.Model):
    english_key = models.CharField(max_length=200)
    translated_entry = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ['english_key', 'translated_entry']
        verbose_name_plural = 'glossary entries'
    
    def __str__(self) -> str:
        return self.as_txt()

    def as_txt(self):
        translated_entry_no_markdown = self.translated_entry.replace('_', '').replace('*', '')
        return f"{self.english_key} -> {translated_entry_no_markdown}"

    def as_dict(self) -> dict:
        return {
            'en': self.english_key,
            'tgt': self.translated_entry,
        }
    
    @classmethod
    def get_entries(cls, sentence: str) -> List['GlossaryEntry']:
        doc = nlp(sentence)
        tokens = [token.lemma_.lower() for token in doc]
        bigrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
        bigrams_nolemma = [f"{token.text.lower()} {token.nbor().text.lower()}" for token in doc[:-1]]
        entries = cls.objects.filter(english_key__in=(tokens + bigrams + bigrams_nolemma)).distinct()
        return list(entries)


class CorpusEntry(models.Model):
    english_text = models.TextField()
    translated_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    source = models.CharField(max_length=200)

    @property
    def tokens(self):
        text = re.sub(r'[^\w\s]', '', self.english_text)
        doc = nlp(text)
        return [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space and not token.is_stop]
    
    def as_txt(self):
        config = SystemConfiguration.load()
        return f"English: {self.english_text}\n{config.target_language_name}: {self.translated_text}"

    def as_dict(self):
        return {
            'en': self.english_text,
            'tgt': self.translated_text,
        }
    
    def __str__(self) -> str:
        return self.as_txt()

    @classmethod
    def init_tantivy_index(cls, lines: List['CorpusEntry']) -> tantivy.Index:
        """ Initialize the Tantivy BM25 index for all corpus entries """
        schema_builder = tantivy.SchemaBuilder()
        schema_builder.add_text_field("text", stored=True, tokenizer_name='en_stem')
        schema_builder.add_text_field("translated_text", stored=True)
        schema_builder.add_integer_field("id", stored=True)  # Add ID field
        schema = schema_builder.build()

        index = tantivy.Index(schema)

        writer = index.writer(num_threads=1)
        for i, line in enumerate(lines):
            writer.add_document(tantivy.Document(
                text=line.english_text,
                tgt_text=line.translated_text,
                id=i  # Store the index
            ))
        writer.commit()
        writer.wait_merging_threads()

        return index
    
    @classmethod
    def get_top_similar_bm25(cls, sent: str, top_n: int = 10) -> List['CorpusEntry']:
        lines = cls.objects.all()

        index = cls.init_tantivy_index(lines)
        searcher = index.searcher()
        query_tokens = cls(english_text=sent, translated_text=None).tokens
        query = index.parse_query(' '.join(query_tokens), ["text"])
        search_results = searcher.search(query, top_n).hits
        if not search_results:
            # race condition: the index is not ready yet
            print("Tantivy: Index not ready, retrying...")
            time.sleep(0.5)
            searcher = index.searcher()
            query_tokens = cls(english_text=sent, translated_text=None).tokens
            query = index.parse_query(' '.join(query_tokens), ["text"])
            search_results = searcher.search(query, top_n).hits

        results = []
        for score, doc_address in search_results:
            doc = searcher.doc(doc_address)
            # Use doc ID to get the relevant CorpusEntry
            doc_id = doc["id"][0]
            results.append(lines[doc_id])

        return results

    class Meta:
        verbose_name = 'translation memory'
        verbose_name_plural = 'translation memories'
        unique_together = ['english_text', 'translated_text']


class Translation(models.Model):
    source_text = models.TextField()
    mt_translation = models.TextField()
    final_translation = models.TextField()
    glossary_entries = models.ManyToManyField(GlossaryEntry, blank=True, help_text='Relevant glossary entries')
    corpus_entries = models.ManyToManyField(CorpusEntry, blank=True, help_text='Relevant corpus entries')
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(get_user_model(), on_delete=models.SET_NULL, null=True)

    def __str__(self) -> str:
        return f"{self.pk}: Translation of '{self.source_text}'"


class SystemConfigurationLazy(LazyObject):
    def _setup(self):
        config = SystemConfiguration.objects.first()
        if config is None:
            config = SystemConfiguration.objects.create()
        self._wrapped = config

    @classmethod
    def load(cls):
        return cls()

class SystemConfiguration(models.Model):
    site_title = models.CharField(max_length=100, default="Medical English to Tetun Translation", blank=False)
    target_language_name = models.CharField(max_length=100, default="Tetun", blank=False)
    target_language_code = models.CharField(max_length=5, default="tet", blank=False, help_text="ISO 639 language code, to be used in the Google Translate API")
    placeholder = models.CharField(max_length=100, default="Stop the wound from bleeding, then do wound dressing.", blank=False, help_text="Placeholder text for the translation input field")
    translation_prompt = models.TextField(
        default="You are a linguist helping to post-edit translations from English to Tetun. "
                "Candidate translations are provided by Google Translate, and you are asked to "
                "correct them, if necessary, using examples and glossary entries. Only use "
                "examples and glossary entries to correct the translations.",
        help_text="System prompt used for the translation post-editing process"
    )
    post_editing_model = models.CharField(
        max_length=100,
        default='gemini/gemini-2.0-flash',
        help_text="The model used for post-editing translations. Choose among https://docs.litellm.ai/docs/providers"
    )
    translation_model = models.CharField(
        max_length=100,
        default='Google Translate',
        help_text="The model used for generating translations. Can be 'Google Translate', or a model from HuggingFace, e.g. 'Helsinki-NLP/opus-mt-en-tdt'"
    )
    num_sentences_retrieved = models.IntegerField(
        default=5,
        help_text="The number of similar sentences retrieved for post-editing"
    )
    dspy_config = models.JSONField(
        default=None,
        blank=True,
        null=True,
        help_text="Optional: Configuration for the dspy model. If present, this takes precedence over the translation_prompt"
    )
    login_required = models.BooleanField(
        default=True,
        help_text="Require users to log in to access the translation interface."
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "System Configuration"
        verbose_name_plural = "System Configuration"

    def save(self, *args, **kwargs):
        if not self.pk and SystemConfiguration.objects.exists():
            raise ValidationError('There can be only one SystemConfiguration instance')
        super().save(*args, **kwargs)

    @classmethod
    def load(cls) -> 'SystemConfiguration':
        return SystemConfigurationLazy.load()


class EvalRow(models.Model):
    en = models.TextField()
    tgt = models.TextField()

    def as_dict(self):
        return {
            'en': self.en,
            'tgt': self.tgt,
        }