"""
fusion.py
=========
Combine multiple per-frame OCR observations of the SAME vehicle
(tracked via vehicle_id) into a single, higher-confidence plate
reading.

Strategy: confidence-weighted, position-wise character voting.

  - Only observations of the same length are grouped together (mixing
    different lengths would produce garbage at the character level);
    the most frequent length wins.
  - For each character position, each observation "votes" for its
    character, weighted by that observation's OCR confidence.
  - The winning character at each position must reach
    `fusion_min_char_agreement` (share of total weight) to be
    accepted; otherwise it becomes '?' -- we do not hallucinate a
    character just because it's needed to complete a plate.
  - Final confidence blends: mean observation confidence, the
    strength of character-level agreement, and observation count
    (more frames -> more trust), which feeds into the engine's
    overall confidence calculation as "temporal_consistency".

A Kalman filter is intentionally NOT used here for character
inference -- see tracker.py for its (bbox-only) role.
"""

from __future__ import annotations

import logging
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .config import ANPRConfig

logger = logging.getLogger("anpr.fusion")


@dataclass
class FrameObservation:
    text: str
    ocr_confidence: float
    detection_confidence: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class FusionResult:
    final_text: str
    confidence: float
    frames_used: int
    temporal_consistency: float  # 0..1, how much frames agreed
    uncertain_positions: int


class TemporalFusion:
    """
    Maintains a short rolling buffer of observations per vehicle_id and
    produces a fused result on demand.
    """

    def __init__(self, config: ANPRConfig):
        self.config = config
        self._buffers: Dict[str, Deque[FrameObservation]] = defaultdict(
            lambda: deque(maxlen=config.max_frame_buffer_per_vehicle)
        )

    def add_observation(self, vehicle_id: str, observation: FrameObservation) -> None:
        self._expire_stale(vehicle_id)
        self._buffers[vehicle_id].append(observation)

    def get_buffer_size(self, vehicle_id: str) -> int:
        return len(self._buffers.get(vehicle_id, []))

    def reset(self, vehicle_id: str) -> None:
        self._buffers.pop(vehicle_id, None)

    def _expire_stale(self, vehicle_id: str) -> None:
        buf = self._buffers.get(vehicle_id)
        if not buf:
            return
        cutoff = time.time() - self.config.track_expiry_seconds
        while buf and buf[0].timestamp < cutoff:
            buf.popleft()

    def fuse(self, vehicle_id: str) -> Optional[FusionResult]:
        self._expire_stale(vehicle_id)
        observations = list(self._buffers.get(vehicle_id, []))
        if not observations:
            return None

        if len(observations) < self.config.min_frames_for_fusion:
            # Not enough frames yet -- fall back to the single best
            # observation rather than forcing a fusion.
            best = max(observations, key=lambda o: o.ocr_confidence)
            return FusionResult(
                final_text=best.text,
                confidence=best.ocr_confidence,
                frames_used=len(observations),
                temporal_consistency=0.0,
                uncertain_positions=0,
            )

        # Group by text length; pick the length with the highest total
        # confidence-weight support (not just raw count).
        length_weight: Dict[int, float] = defaultdict(float)
        for obs in observations:
            length_weight[len(obs.text)] += obs.ocr_confidence

        target_len = max(length_weight, key=lambda l: length_weight[l])
        grouped = [o for o in observations if len(o.text) == target_len]

        # Position-wise weighted voting.
        position_votes: List[Counter] = [Counter() for _ in range(target_len)]
        for obs in grouped:
            weight = obs.ocr_confidence
            for idx, ch in enumerate(obs.text):
                position_votes[idx][ch] += weight

        final_chars: List[str] = []
        agreement_scores: List[float] = []
        uncertain = 0

        for votes in position_votes:
            total_weight = sum(votes.values())
            if total_weight <= 0:
                final_chars.append("?")
                uncertain += 1
                agreement_scores.append(0.0)
                continue
            char, weight = votes.most_common(1)[0]
            share = weight / total_weight
            agreement_scores.append(share)
            if share >= self.config.fusion_min_char_agreement:
                final_chars.append(char)
            else:
                final_chars.append("?")
                uncertain += 1

        final_text = "".join(final_chars)
        temporal_consistency = sum(agreement_scores) / len(agreement_scores)
        mean_conf = sum(o.ocr_confidence for o in grouped) / len(grouped)

        # Blend mean OCR confidence with agreement strength; more
        # supporting frames (up to a cap) nudges confidence up slightly.
        frame_bonus = min(0.1, 0.02 * (len(grouped) - 1))
        fused_confidence = min(1.0, 0.6 * mean_conf + 0.4 * temporal_consistency + frame_bonus)

        logger.debug(
            "Fused %d/%d observations for vehicle_id=%s -> %r (conf=%.2f, agreement=%.2f)",
            len(grouped), len(observations), vehicle_id, final_text,
            fused_confidence, temporal_consistency,
        )

        return FusionResult(
            final_text=final_text,
            confidence=fused_confidence,
            frames_used=len(grouped),
            temporal_consistency=temporal_consistency,
            uncertain_positions=uncertain,
        )
