import os
import csv
from typing import List, Optional
import subprocess

from dataclasses import dataclass

@dataclass
class Language:
    nllb_code: str
    openwho_code: str
    name: str
    low_res: bool = False
    googletrans_code: Optional[str] = None

    def __str__(self):
        return self.name

class LanguageCollection(list):
    def __init__(self, items: List[Language]):
        super().__init__(items)
    
    def get_by_name(self, name: str) -> Optional[Language]:
        for lang in self:
            if lang.name == name:
                return lang
        return None
    
    def get_by_openwho_code(self, code: str) -> Optional[Language]:
        for lang in self:
            if lang.openwho_code == code:
                return lang
        return None
    
    def get_by_nllb_code(self, code: str) -> Optional[Language]:
        for lang in self:
            if lang.nllb_code == code:
                return lang
        return None


languages = LanguageCollection([
    Language("eng_Latn", "en", "English"),
    # institutionalized, low-res, part of Flores, part of Gatitos
    Language("tpi_Latn", "tpi", "Tok Pisin", low_res=True, googletrans_code="tpi"),
    Language("dzo_Tibt", "dz", "Dzongkha", low_res=True, googletrans_code="dz"),  # Bhutan
    Language("quy_Latn", "qu", "Quechua", low_res=True, googletrans_code="qu"),  # Peru, however Gatitos has Quechua, and this is Ayacucho Quechua, maybe different
    Language("run_Latn", "rn", "Kirundi", low_res=True, googletrans_code="rn"),  # Burundi
    Language("lin_Latn", "ln", "Lingala", low_res=True, googletrans_code="ln"),  # Congo
    Language("asm_Beng", "as", "Assamese", low_res=True, googletrans_code="as"),  # India
])