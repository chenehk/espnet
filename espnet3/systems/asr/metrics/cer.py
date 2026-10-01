"""Character error rate metric utilities."""

from __future__ import annotations

from typing import List

from espnet3.systems.asr.metrics.base_error_rate import BaseErrorRate

#: What espnet2's CharTokenizer emits for a space, and therefore what
#: ``asr.sh`` scores CER over. It is one token either way, so this changes how
#: an alignment reads rather than what the denominator counts.
SPACE_SYMBOL = "<space>"


class CER(BaseErrorRate):
    """Compute CER for a dataset.

    Characters include spaces, which is what ``jiwer.cer`` counted and what
    ``asr.sh`` counts via ``CharTokenizer``. See
    :mod:`espnet3.systems.asr.metrics.base_error_rate` for the two behaviours that
    differ from the previous ``jiwer`` implementation.
    """

    metric_name = "CER"
    alignment_filename = "cer_alignment"

    def tokenize(self, text: str) -> List[str]:
        """Split into characters, spaces included."""
        return [SPACE_SYMBOL if char == " " else char for char in text]
