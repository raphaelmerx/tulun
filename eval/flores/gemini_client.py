import csv
import os
import json
import hashlib
from typing import List, Tuple, Literal
from dataclasses import dataclass
from dotenv import load_dotenv 
load_dotenv()
from typing import Optional

from flores_utils import CorpusEntry

import google.generativeai as genai
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

thisdir = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Message:
    role: Literal['user', 'system', 'assistant']
    content: str

    @staticmethod
    def format_query(line: CorpusEntry, machine_translated: str) -> 'Message':
        return Message(role='user', content=f"<English>{line.src_text}\n<{line.tgt_lang} MT>{machine_translated}</{line.tgt_lang} MT>")
    
    @staticmethod
    def format_response(line: CorpusEntry) -> 'Message':
        return Message(role='assistant', content=f"<{line.tgt_lang} (post-edited)>{line.tgt_text}</{line.tgt_lang} (post-edited)>")
    
    def to_dict(self):
        return {
            'role': self.role,
            'content': self.content,
        }
    
    def __str__(self):
        return f"{self.role}: {self.content}"
    

class GenerativeModel:
    def __init__(self, client: Optional[genai.GenerativeModel] = None, **kwargs):
        if not client:
            self.client = genai.GenerativeModel('gemini-2.0-flash', **kwargs)
        else:
            self.client = client
        self.load_memory()

    def load_memory(self, filename='gemini_memory.json'):
        filepath = os.path.join(thisdir, 'datafiles', filename)
        if not os.path.exists(filepath):
            self._memory = {}
            return
        with open(filepath, 'r') as f:
            self._memory = json.load(f)
        print(f"Gemini client: Loaded {len(self._memory)} memory entries")
    
    def save_memory(self, filename='gemini_memory.json'):
        filepath = os.path.join(thisdir, 'datafiles', filename)
        with open(filepath, 'w') as f:
            json.dump(self._memory, f, indent=2)
    
    def get_response(self, messages: Tuple[Message]) -> str:
        memory_key = hashlib.md5(json.dumps([m.to_dict() for m in messages]).encode()).hexdigest()
        if memory_key in self._memory:
            return self._memory[memory_key]

        formatted_messages = self.format_messages_for_gemini(messages)
        response = self.client.generate_content(
            formatted_messages,
        )
        result = response.text.strip()
        self._memory[memory_key] = result
        self.save_memory()
        return result

    @staticmethod
    def format_messages_for_gemini(messages: Tuple[Message]) -> List[dict]:
        return [{
            "role": "user" if message.role == "user" else "model",
            "parts": [message.content]
        } for message in messages if message.role != 'system']