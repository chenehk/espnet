"""Token error rate metric utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet3.systems.asr.metrics.base_error_rate import BaseErrorRate


class TER(BaseErrorRate):
    """Compute TER (token error rate) for a dataset.

    TER is the error rate over the model's subword (BPE) tokens: the reference
    and hypothesis are tokenized with a SentencePiece model, then scored like
    WER over the resulting token sequences. This mirrors espnet2's Stage 13
    scoring, which computes ``ter`` at the ``bpe`` token level.

    Note that ``asr.sh`` passes no non-linguistic symbols on the ``bpe`` path,
    so nothing is removed before tokenization here either. See
    :mod:`espnet3.systems.asr.metrics.base_error_rate` for the two behaviours that
    differ from the previous ``jiwer`` implementation.
    """

    metric_name = "TER"
    alignment_filename = "ter_alignment"

    def __init__(
        self,
        bpemodel: str | Path,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Optional[Iterable[str]] = None,
        case: str = "fold",
        costs: str = "sclite",
    ) -> None:
        """Initialize the TER metric.

        Args:
            bpemodel: Path to the SentencePiece model used to tokenize text
                into subword tokens (typically the recipe's ``bpe.model``).
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Cleaner names, as espnet2's TextCleaner takes them.
            case: ``"fold"`` compares case-insensitively, as ``sclite`` does.
            costs: Alignment cost model, ``"sclite"`` or ``"unit"``.
        """
        super().__init__(
            ref_key=ref_key,
            hyp_key=hyp_key,
            clean_types=clean_types,
            case=case,
            costs=costs,
        )
        self.tokenizer = SentencepiecesTokenizer(bpemodel)

    def tokenize(self, text: str) -> List[str]:
        """Split into SentencePiece subword tokens."""
        return list(self.tokenizer.text2tokens(text))
