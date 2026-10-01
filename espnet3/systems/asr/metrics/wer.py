"""Word error rate metric utilities."""

from __future__ import annotations

from typing import List

from espnet3.systems.asr.metrics.base_error_rate import BaseErrorRate


class WER(BaseErrorRate):
    """Compute WER for hypotheses.

    Words are whitespace-delimited, and the corpus figure is
    ``sum(errors) / sum(reference words)``, matching what ``sclite`` reports
    for the same text. See :mod:`espnet3.systems.asr.metrics.base_error_rate` for
    the two behaviours that differ from the previous ``jiwer`` implementation.
    """

    metric_name = "WER"
    alignment_filename = "wer_alignment"

    def tokenize(self, text: str) -> List[str]:
        """Split into whitespace-delimited words."""
        return text.split()
