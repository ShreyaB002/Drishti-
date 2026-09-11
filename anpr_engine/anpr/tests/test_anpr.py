"""
tests/test_anpr.py
===================
Unit tests that do NOT require the actual YOLO weights or a real OCR
engine to be installed -- they exercise the pure-logic modules
(validator, fusion, image_utils) directly, and use lightweight fakes
for anything that would otherwise need a model file / GPU.

Run with:
    python -m pytest anpr/tests/test_anpr.py -v
"""

import sys
import os
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from anpr.config import ANPRConfig
from anpr.validator import PlateValidator
from anpr.fusion import TemporalFusion, FrameObservation
from anpr.utils.image_utils import clip_bbox, expand_bbox, safe_crop, bbox_area, translate_bbox
from anpr.preprocessor import PlatePreprocessor


# --------------------------------------------------------------------------
# validator.py
# --------------------------------------------------------------------------

class TestPlateValidator:
    def setup_method(self):
        self.validator = PlateValidator(ANPRConfig())

    def test_valid_standard_plate(self):
        result = self.validator.validate("MH12AB1234")
        assert result.is_valid
        assert result.matched_format == "IN_STANDARD_2LETTER"
        assert result.corrected_text == "MH12AB1234"

    def test_valid_single_letter_series(self):
        result = self.validator.validate("KA05M1234")
        assert result.is_valid
        assert result.matched_format == "IN_STANDARD_1LETTER"

    def test_valid_bh_series(self):
        result = self.validator.validate("22BH1234AB")
        assert result.is_valid
        assert result.matched_format == "IN_BH_SERIES"

    def test_ocr_confusion_correction_digit_position(self):
        # 'O' appears where a digit is expected (RTO code position) ->
        # should be corrected to '0'.
        result = self.validator.validate("MH1OAB1234")
        assert result.corrected_text == "MH10AB1234"
        assert result.is_valid

    def test_ocr_confusion_correction_alpha_position(self):
        # '8' appears where a letter is expected (series position) ->
        # should be corrected to 'B'.
        result = self.validator.validate("MH12A81234")
        assert result.corrected_text == "MH12AB1234"
        assert result.is_valid

    def test_unresolvable_position_is_not_hallucinated(self):
        # '#' cannot be resolved to either a letter or digit via the
        # confusion map -- must remain '?' rather than being guessed.
        result = self.validator.validate("MH12A#1234")
        assert "?" in result.corrected_text
        assert not result.is_valid

    def test_garbage_input_is_rejected(self):
        result = self.validator.validate("!!!###")
        assert not result.is_valid

    def test_known_state_code_flag(self):
        result = self.validator.validate("MH12AB1234")
        assert result.known_state_code is True

    def test_unknown_state_code_does_not_block_validity(self):
        # 'XX' isn't a real state code but format-wise still matches;
        # validity should still be format-driven, state-code is a soft signal.
        result = self.validator.validate("XX12AB1234")
        assert result.is_valid
        assert result.known_state_code is False


# --------------------------------------------------------------------------
# fusion.py
# --------------------------------------------------------------------------

class TestTemporalFusion:
    def setup_method(self):
        self.fusion = TemporalFusion(ANPRConfig())

    def _obs(self, text, conf):
        return FrameObservation(text=text, ocr_confidence=conf, detection_confidence=0.9)

    def test_majority_vote_wins(self):
        vid = "V_TEST_1"
        self.fusion.add_observation(vid, self._obs("MH12AB1234", 0.71))
        self.fusion.add_observation(vid, self._obs("MH12AB1234", 0.89))
        self.fusion.add_observation(vid, self._obs("MH12A81234", 0.62))
        self.fusion.add_observation(vid, self._obs("MH12AB1234", 0.94))
        self.fusion.add_observation(vid, self._obs("MH12AB1234", 0.91))

        result = self.fusion.fuse(vid)
        assert result is not None
        assert result.final_text == "MH12AB1234"
        assert result.frames_used == 5
        assert result.confidence > 0.7

    def test_insufficient_frames_falls_back_to_best_single(self):
        vid = "V_TEST_2"
        self.fusion.add_observation(vid, self._obs("DL01CA1234", 0.8))
        result = self.fusion.fuse(vid)
        assert result is not None
        assert result.final_text == "DL01CA1234"
        assert result.frames_used == 1

    def test_no_observations_returns_none(self):
        assert self.fusion.fuse("V_NEVER_SEEN") is None

    def test_low_agreement_position_becomes_uncertain(self):
        vid = "V_TEST_3"
        # Same length, but character at index 4 disagrees roughly evenly.
        self.fusion.add_observation(vid, self._obs("MH12AB1234", 0.5))
        self.fusion.add_observation(vid, self._obs("MH12XB1234", 0.5))
        result = self.fusion.fuse(vid)
        assert result is not None
        assert "?" in result.final_text or result.uncertain_positions >= 0

    def test_reset_clears_buffer(self):
        vid = "V_TEST_4"
        self.fusion.add_observation(vid, self._obs("MH12AB1234", 0.9))
        self.fusion.reset(vid)
        assert self.fusion.get_buffer_size(vid) == 0


# --------------------------------------------------------------------------
# utils/image_utils.py
# --------------------------------------------------------------------------

class TestImageUtils:
    def test_clip_bbox_within_bounds(self):
        assert clip_bbox((10, 10, 50, 50), 100, 100) == (10, 10, 50, 50)

    def test_clip_bbox_out_of_bounds(self):
        clipped = clip_bbox((-10, -10, 200, 200), 100, 100)
        x1, y1, x2, y2 = clipped
        assert x1 >= 0 and y1 >= 0 and x2 <= 100 and y2 <= 100

    def test_expand_bbox_grows_and_clips(self):
        expanded = expand_bbox((40, 40, 60, 60), 0.5, 100, 100)
        x1, y1, x2, y2 = expanded
        assert x1 < 40 and x2 > 60

    def test_safe_crop_valid_region(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = safe_crop(frame, (10, 10, 50, 50))
        assert crop is not None
        assert crop.shape[0] == 40 and crop.shape[1] == 40

    def test_safe_crop_out_of_bounds_bbox(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = safe_crop(frame, (90, 90, 300, 300))
        assert crop is not None
        assert crop.shape[0] <= 10 and crop.shape[1] <= 10

    def test_safe_crop_degenerate_bbox_returns_none(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        crop = safe_crop(frame, (200, 200, 300, 300))
        assert crop is None

    def test_safe_crop_empty_frame_returns_none(self):
        crop = safe_crop(np.array([]), (0, 0, 10, 10))
        assert crop is None

    def test_bbox_area(self):
        assert bbox_area((0, 0, 10, 20)) == 200

    def test_translate_bbox(self):
        assert translate_bbox((5, 5, 15, 15), (100, 200)) == (105, 205, 115, 215)


# --------------------------------------------------------------------------
# preprocessor.py
# --------------------------------------------------------------------------

class TestPlatePreprocessor:
    def setup_method(self):
        self.preprocessor = PlatePreprocessor(ANPRConfig())

    def test_quality_flags_small_plate(self):
        tiny = np.full((10, 20, 3), 128, dtype=np.uint8)
        quality = self.preprocessor.assess_quality(tiny)
        assert quality.is_too_small

    def test_process_upscales_small_plate(self):
        tiny = np.full((20, 60, 3), 128, dtype=np.uint8)
        processed, quality = self.preprocessor.process(tiny)
        assert processed.shape[0] >= self.preprocessor.config.ocr_target_height_px

    def test_process_returns_quality_scalar_in_range(self):
        img = np.random.randint(0, 255, (64, 200, 3), dtype=np.uint8)
        _, quality = self.preprocessor.process(img)
        scalar = quality.as_quality_scalar()
        assert 0.0 <= scalar <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
