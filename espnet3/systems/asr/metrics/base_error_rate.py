"""Shared base for the error-rate metrics, which are adapters over SPLET.

The scoring lives in ``splet/``, which knows nothing about ESPnet and
reproduces ``sclite`` exactly (espnet/espnet#6760). What is left here is the
part that is ESPnet's: reading the SCP files ``measure`` hands over, writing
the alignment file next to them, and returning the keys ``metrics.json``
expects.

Two behaviours changed when this stopped calling ``jiwer``, both deliberate.
Neither is invisible, so both are stated here with what they move:

* **Case.** ``sclite`` compares case-insensitively unless it is given ``-s``,
  and only four egs2 corpora pass it, so ``case="fold"`` is the default here
  as it is in SPLET. ``jiwer`` was case-sensitive. This matters whenever
  *either* side carries case, not only the reference: a model that emits
  cased text against a lower-case reference was being charged a substitution
  per word for it. On ``egs3/owsm_v4``'s MLS_en_test that is the difference
  between 28.95 and 23.05 WER. The four spgispeech runs, lower-case on both
  sides, do not move at all. Pass ``case="sensitive"`` for the old
  comparison.
* **Empty hypotheses.** The previous code substituted ``"."`` for an empty
  string on both sides. An undecodable utterance then scored one substitution
  instead of a deletion per reference word, which flatters the system; #6735
  removed the same placeholder from the BLEU metric for the same reason. The
  reference length is now the reference's own length. No egs3 ASR output has
  an empty hypothesis today, so nothing moves on what is in the repository.

``sclite``'s totals are an upper bound on ``jiwer``'s rather than equal to
them, because it minimises a weighted cost rather than an edit count. The two
agree on clean output and separate as the error rate rises. See
:mod:`splet.alignment`.
"""

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from espnet3.components.metrics.base_metric import BaseMetric
from splet.alignment import levenshtein_alignment
from splet.normalizers import build_normalizer

# espnet2's TextCleaner names, mapped onto SPLET's normalizers. Only the two
# Whisper ones can change an egs2 ASR score -- they are the only cleaners any
# ASR-family recipe sets -- and they are the only ones SPLET implements.
_CLEANER_TO_NORMALIZER = {
    "whisper_en": {"name": "whisper", "language": "en"},
    "whisper_basic": {"name": "whisper", "language": "basic"},
}


def normalizer_for(clean_types: Optional[Iterable[str]]):
    """Build a SPLET normalizer from espnet2 cleaner names.

    Args:
        clean_types: Cleaner names as ``asr.sh`` would pass them, or None.

    Returns:
        A SPLET normalizer, the identity when nothing was asked for.

    Raises:
        NotImplementedError: For a cleaner SPLET does not implement. Silently
            skipping it would report a score under a normalization that never
            ran.
    """
    names = list(clean_types or [])
    unsupported = [name for name in names if name not in _CLEANER_TO_NORMALIZER]
    if unsupported:
        raise NotImplementedError(
            f"SPLET does not implement the {unsupported} text cleaner(s). "
            f"Available: {sorted(_CLEANER_TO_NORMALIZER)}. These are the only "
            "cleaners any egs2 ASR-family recipe sets, so the others have not "
            "been ported; see espnet/espnet#6760."
        )
    return build_normalizer([_CLEANER_TO_NORMALIZER[name] for name in names] or None)


class BaseErrorRate(BaseMetric):
    """An error rate over one tokenization, computed by SPLET.

    Subclasses set :attr:`metric_name`, :attr:`tokenize` and
    :attr:`alignment_filename`.
    """

    #: Key this metric reports under, and the prefix for its counts.
    metric_name: str = ""
    #: File the per-utterance alignment is written to.
    alignment_filename: str = ""

    def __init__(
        self,
        ref_key: str = "ref",
        hyp_key: str = "hyp",
        clean_types: Optional[Iterable[str]] = None,
        case: str = "fold",
        costs: str = "sclite",
    ) -> None:
        """Initialize the metric.

        Args:
            ref_key: Key name for reference text entries.
            hyp_key: Key name for hypothesis text entries.
            clean_types: Cleaner names, as espnet2's TextCleaner takes them.
            case: ``"fold"`` compares case-insensitively, as ``sclite`` does
                by default. ``"sensitive"`` is ``sclite``'s ``-s``, and is
                what ``jiwer`` did.
            costs: ``"sclite"`` reproduces sclite's weighted alignment;
                ``"unit"`` is plain Levenshtein, which is what ``jiwer``
                computes.
        """
        self.ref_key = ref_key
        self.hyp_key = hyp_key
        self.clean_types = list(clean_types or [])
        self.normalizer = normalizer_for(clean_types)
        self.case = case
        self.costs = costs

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Split normalized text into the units this metric counts."""
        raise NotImplementedError

    def _prepare(self, text: str) -> List[str]:
        """Normalize, fold case if asked, and tokenize."""
        text = self.normalizer(text)
        if self.case == "fold":
            text = text.lower()
        return self.tokenize(text)

    def __call__(
        self,
        data: Dict[str, Path],
        test_name: str,
        inference_dir: Path,
    ) -> Dict[str, Any]:
        """Score a test set, write its alignment, and return the counts.

        Args:
            data: Mapping of input aliases to SCP paths. ``data[ref_key]`` and
                ``data[hyp_key]`` must share utterance IDs in the same order.
            test_name: Test set name, used for the output directory.
            inference_dir: Directory the test set's outputs live in.

        Returns:
            The rate as a percentage under :attr:`metric_name`, plus the
            counts it was pooled from: ``_errors``, ``_ref_len``, ``_sub``,
            ``_del``, ``_ins`` and ``_hit``. The rate is
            ``sum(errors) / sum(ref_len)``, which is what SCTK reports and is
            not the mean of the per-utterance rates.
        """
        name = self.metric_name
        errors = ref_len = substitutions = deletions = insertions = hits = 0
        rendered = []

        for utt_id, row in self.iter_inputs(data, self.ref_key, self.hyp_key):
            alignment = levenshtein_alignment(
                self._prepare(row[self.ref_key]),
                self._prepare(row[self.hyp_key]),
                costs=self.costs,
            )
            errors += alignment.errors
            ref_len += alignment.ref_len
            substitutions += alignment.substitutions
            deletions += alignment.deletions
            insertions += alignment.insertions
            hits += alignment.hits
            rendered.append(f"{utt_id}\n{alignment.to_string()}\n")

        test_dir = Path(inference_dir) / test_name
        test_dir.mkdir(parents=True, exist_ok=True)
        with (test_dir / self.alignment_filename).open("w", encoding="utf-8") as f:
            f.write("\n".join(rendered))

        return {
            name: round(100 * errors / ref_len, 2) if ref_len else 0.0,
            f"{name}_errors": errors,
            f"{name}_ref_len": ref_len,
            f"{name}_sub": substitutions,
            f"{name}_del": deletions,
            f"{name}_ins": insertions,
            f"{name}_hit": hits,
        }
