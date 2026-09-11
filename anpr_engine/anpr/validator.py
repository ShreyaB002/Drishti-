"""
validator.py
============
Configurable validation of OCR'd text against Indian number-plate
formats, plus position-aware character correction.

Two distinct jobs live here:

1. `validate(text)` -- does the cleaned text match any configured
   PlateFormat? Returns the matched format (if any).

2. `correct_plate(text)` -- for near-miss strings, try swapping
   commonly-confused characters (O/0, I/1, B/8, S/5, Z/2, ...) but
   ONLY at positions where the target format's char_mask says a
   different character class is expected. This is a conditional,
   format-aware correction -- never a blind global substitution --
   and it never invents characters that weren't in the OCR output;
   unresolvable positions are left as '?'.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import ANPRConfig, PlateFormat, OCR_CONFUSION_MAP, INDIAN_STATE_CODES

logger = logging.getLogger("anpr.validator")

_ALPHA_TARGETS = {v: k for k, v in OCR_CONFUSION_MAP.items() if k.isdigit() and v.isalpha()}


@dataclass
class ValidationResult:
    is_valid: bool
    matched_format: Optional[str]
    corrected_text: str
    known_state_code: bool
    uncertain_positions: int  # count of '?' placeholders left in corrected_text


class PlateValidator:
    def __init__(self, config: ANPRConfig):
        self.config = config
        self._compiled = [
            (fmt, re.compile(fmt.pattern)) for fmt in config.plate_formats
        ]

    def validate(self, text: str) -> ValidationResult:
        """
        Try direct validation first; if nothing matches, attempt
        position-aware correction against each candidate format and
        re-validate the corrected string.
        """
        text = text.strip().upper()

        direct = self._match_any(text)
        if direct is not None:
            return ValidationResult(
                is_valid=True,
                matched_format=direct.name,
                corrected_text=text,
                known_state_code=self._is_known_state(text),
                uncertain_positions=0,
            )

        # No direct match -- try format-aware correction against the
        # closest-length format(s).
        best: Optional[Tuple[PlateFormat, str, int]] = None  # (format, corrected, unresolved)
        for fmt in self.config.plate_formats:
            if len(text) != len(fmt.char_mask):
                continue
            corrected, unresolved = self._correct_against_mask(text, fmt.char_mask)
            if best is None or unresolved < best[2]:
                best = (fmt, corrected, unresolved)

        if best is None:
            return ValidationResult(
                is_valid=False,
                matched_format=None,
                corrected_text=text,
                known_state_code=self._is_known_state(text),
                uncertain_positions=0,
            )

        fmt, corrected, unresolved = best
        # Only claim a format match if every position could be resolved
        # to a concrete character (no '?' placeholders) AND the regex
        # actually matches.
        is_fully_resolved = "?" not in corrected
        matched = is_fully_resolved and self._compiled_match(fmt, corrected)

        return ValidationResult(
            is_valid=bool(matched),
            matched_format=fmt.name if matched else None,
            corrected_text=corrected,
            known_state_code=self._is_known_state(corrected),
            uncertain_positions=unresolved,
        )

    def correct_plate(self, text: str) -> str:
        """
        Public convenience wrapper: return the best-effort corrected
        string (may still contain '?' for unresolved positions), even
        if it does not end up matching a known format.
        """
        return self.validate(text).corrected_text

    # -- internals -----------------------------------------------------

    def _match_any(self, text: str) -> Optional[PlateFormat]:
        for fmt, pattern in self._compiled:
            if pattern.match(text):
                return fmt
        return None

    def _compiled_match(self, fmt: PlateFormat, text: str) -> bool:
        for f, pattern in self._compiled:
            if f.name == fmt.name:
                return bool(pattern.match(text))
        return False

    def _is_known_state(self, text: str) -> bool:
        return len(text) >= 2 and text[:2] in INDIAN_STATE_CODES

    def _correct_against_mask(self, text: str, mask: str) -> Tuple[str, int]:
        """
        Walk the string position by position. Where the observed
        character's class (alpha/digit) doesn't match what the mask
        expects, attempt a known-confusion swap. If no valid swap
        exists, leave a '?' placeholder rather than guessing.
        """
        out_chars: List[str] = []
        unresolved = 0

        for ch, expected in zip(text, mask):
            expected_is_alpha = expected == "A"
            ch_is_alpha = ch.isalpha()

            if ch_is_alpha == expected_is_alpha:
                out_chars.append(ch)
                continue

            swapped = OCR_CONFUSION_MAP.get(ch)
            if swapped is not None and swapped.isalpha() == expected_is_alpha:
                out_chars.append(swapped)
            else:
                out_chars.append("?")
                unresolved += 1

        return "".join(out_chars), unresolved
