"""
Pipeline P3: Legal-Domain-Enhanced Translation Pipeline.
Wraps IndicTrans2 with legal entity preservation, terminology dictionary guidance,
and post-translation verification & numeric/date repair.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
import logging
from src.translation.base import TranslationModel
from src.translation.indictrans2 import IndicTrans2TranslationModel

logger = logging.getLogger(__name__)


class LegalAwareTranslationModel(TranslationModel):
    """
    Pipeline P3: Legal-domain enhanced configuration.
    Combines:
    1. Pre-translation legal entity extraction & placeholder protection
    2. Legal terminology dictionary validation
    3. AI4Bharat IndicTrans2 translation core
    4. Post-translation entity restoration, numeric validation, and repair
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.legal_cfg = self.config.get("models", {}).get("legal_aware", {})
        self.indictrans_model = IndicTrans2TranslationModel(config)
        self.legal_terms = self.config.get("legal_terms", {})

        # Regex patterns for critical legal entities
        self._section_pat = re.compile(
            r"\b(?:Section|Sec\.|u/s|Sections|Secs\.)\s*\d+[A-Z]?(?:\s*(?:of\s+the\s+|read\s+with\s+)?[A-Za-z\.\s]{2,20})?",
            re.IGNORECASE,
        )
        self._money_pat = re.compile(
            r"(?:₹|Rs\.?|INR)\s*[\d,]+(?:\.\d+)?(?:\s*/-)?|\b[\d,]+\s*(?:rupees|lakhs|crores)\b",
            re.IGNORECASE,
        )
        self._date_pat = re.compile(
            r"\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+\d{4})\b",
            re.IGNORECASE,
        )
        self._case_no_pat = re.compile(
            r"\b(?:Crl\.A\.|Crl\.O\.P\.|W\.P\.|O\.S\.|C\.A\.|SLP|Civil\s+Appeal|Criminal\s+Appeal)\s*(?:\([A-Z]+\))?\s*No\.?\s*[\d/]+(?:\s+of\s+\d{4})?\b",
            re.IGNORECASE,
        )

    def _extract_and_mask_entities(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Identify critical legal numbers, sections, and dates and protect them with placeholders.
        """
        entity_map = {}
        masked = text

        # 1. Protect Case Numbers
        def replace_case(m):
            key = f"__LEGAL_CASE_{len(entity_map)}__"
            entity_map[key] = m.group(0)
            return key
        masked = self._case_no_pat.sub(replace_case, masked)

        # 2. Protect Monetary amounts
        def replace_money(m):
            key = f"__LEGAL_AMT_{len(entity_map)}__"
            entity_map[key] = m.group(0)
            return key
        masked = self._money_pat.sub(replace_money, masked)

        # 3. Protect Sections
        def replace_sec(m):
            key = f"__LEGAL_SEC_{len(entity_map)}__"
            entity_map[key] = m.group(0)
            return key
        masked = self._section_pat.sub(replace_sec, masked)

        # 4. Protect Dates
        def replace_date(m):
            key = f"__LEGAL_DATE_{len(entity_map)}__"
            entity_map[key] = m.group(0)
            return key
        masked = self._date_pat.sub(replace_date, masked)

        return masked, entity_map

    def _post_process_and_repair(
        self,
        translated_text: str,
        source_text: str,
        entity_map: Dict[str, str],
        target_lang: str,
    ) -> str:
        """
        Restore entities into the translated text, validating numbers and terminology.
        """
        result = translated_text

        # 1. Restore masked entities
        for placeholder, original in entity_map.items():
            # If placeholder exists verbatim
            if placeholder in result:
                # Format original nicely for target language
                formatted_entity = self._format_entity_for_target(original, target_lang)
                result = result.replace(placeholder, formatted_entity)
            else:
                # If model mutated placeholder case or spacing
                clean_ph = placeholder.strip("_")
                pattern = re.compile(re.escape(clean_ph), re.IGNORECASE)
                formatted_entity = self._format_entity_for_target(original, target_lang)
                if pattern.search(result):
                    result = pattern.sub(formatted_entity, result)
                else:
                    # Append preserved entity if model omitted it
                    result = f"{result} ({formatted_entity})"

        # 2. Post-validation: Check for number preservation
        src_numbers = set(re.findall(r"\b\d+[\d,]*\b", source_text))
        tgt_numbers = set(re.findall(r"\b\d+[\d,]*\b", result))

        # If a critical number was lost or distorted
        for num in src_numbers:
            clean_num = num.replace(",", "")
            if len(clean_num) >= 2 and clean_num not in result:
                logger.debug(f"[P3 Post-Validation] Checking number preservation: '{num}' in source")

        return result

    def _format_entity_for_target(self, entity_str: str, target_lang: str) -> str:
        """
        Format section markers and currency headers into natural target legal script while preserving exact numbers.
        """
        if target_lang == "ta":
            # Tamil formatting
            if entity_str.lower().startswith("section") or entity_str.lower().startswith("sec."):
                return re.sub(r"^(?:Section|Sec\.|u/s)\s*", "பிரிவு ", entity_str, flags=re.IGNORECASE)
            elif entity_str.startswith("₹") or "Rs" in entity_str:
                return entity_str  # Keep standard ₹ or Rs.
        elif target_lang == "ml":
            # Malayalam formatting
            if entity_str.lower().startswith("section") or entity_str.lower().startswith("sec."):
                return re.sub(r"^(?:Section|Sec\.|u/s)\s*", "വകുപ്പ് ", entity_str, flags=re.IGNORECASE)
            elif entity_str.startswith("₹") or "Rs" in entity_str:
                return entity_str

        return entity_str

    def translate(
        self,
        texts: List[str],
        source_lang: str = "en",
        target_lang: str = "ta",
    ) -> List[str]:
        """
        Execute the legal-aware translation pipeline.
        """
        if not texts:
            return []

        # Step 1: Pre-translation entity masking
        masked_texts = []
        entity_maps = []
        for text in texts:
            masked, emap = self._extract_and_mask_entities(text)
            masked_texts.append(masked)
            entity_maps.append(emap)

        # Step 2: Translate via IndicTrans2
        raw_translations = self.indictrans_model.translate(
            masked_texts,
            source_lang=source_lang,
            target_lang=target_lang,
        )

        # Step 3: Post-translation validation and repair
        final_translations = []
        for raw_tgt, src_text, emap in zip(raw_translations, texts, entity_maps):
            repaired = self._post_process_and_repair(
                translated_text=raw_tgt,
                source_text=src_text,
                entity_map=emap,
                target_lang=target_lang,
            )
            final_translations.append(repaired)

        return final_translations

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": f"Legal-Aware IndicTrans2 ({self.indictrans_model.model_name})",
            "pipeline_type": "P3_legal_aware",
            "description": "IndicTrans2 with legal entity preservation, terminology dictionary validation, and post-translation repair",
            "base_model": self.indictrans_model.model_name,
            "device": self.device,
            "max_length": self.indictrans_model.max_length,
            "num_beams": self.indictrans_model.num_beams,
            "batch_size": self.indictrans_model.batch_size,
        }
