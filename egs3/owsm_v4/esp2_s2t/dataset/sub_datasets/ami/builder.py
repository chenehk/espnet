"""AMI as OWSM utterances.

There is no ``prepare_ami.py``: ``egs2/owsm_v3/s2t1/local/prepare_ami.sh`` runs
``egs2/ami/asr1/local/data.sh`` and then
``egs2/owsm_v3/s2t1/local/kaldi_to_whisper.py``. Both layers are reproduced
here against the raw NXT annotations, so neither the Java exporter that
``local/ami_xml2text.sh`` calls nor the Kaldi directory is materialised.

One headset channel of one meeting is one recording. Its transcriber segments
are cut into utterances by ``local/ami_split_segments.pl`` -- reimplemented
below, quirks included -- and grouped back by recording so
``generate_long_utterances`` packs them into 30 s spans.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable, Iterator

from egs3.owsm_v4.esp2_s2t.dataset.builder import OWSMBuilder
from egs3.owsm_v4.esp2_s2t.dataset.utils import (
    Utterance,
    generate_long_utterances,
    sub_dataset_config,
)

_CONFIG = sub_dataset_config(__name__)
_BUILDER = _CONFIG["builder"]

LANG = str(_CONFIG["dataset"]["lang"])
TASK = str(_CONFIG["dataset"]["task"])
ANNOTATIONS_SUBDIR = str(_BUILDER["annotations_subdir"])
AUDIO_TEMPLATE = str(_BUILDER["audio_template"])
RECORDING_PREFIX = str(_BUILDER["recording_prefix"])
READ_HEADERS = bool(_BUILDER["read_headers"])
MAX_WORDS = int(_BUILDER["max_words_per_segment"])
MIN_DURATION = float(_BUILDER["min_duration"])
MEETINGS = {
    str(split): [str(x) for x in names]
    for split, names in _BUILDER["meetings"].items()
}

NITE = "{http://nite.sourceforge.net/}"
HREF = re.compile(r"#id\(([^)]*)\)(?:\.\.id\(([^)]*)\))?")
PUNCTUATION = re.compile(r"[.,?!:]")
TIME_TOKEN = re.compile(r"[0-9]+\.[0-9]*")


def read_meetings(annotations: Path) -> dict:
    """Return ``{(meeting, agent): (speaker, channel)}`` from meetings.xml.

    The NXT query behind ``transcripts0`` takes the speaker's ``global_name``
    and ``channel`` from here, not from the segment.
    """
    out = {}
    root = ET.parse(annotations / "corpusResources" / "meetings.xml").getroot()
    for meeting in root:
        name = meeting.get("observation")
        for speaker in meeting:
            out[(name, speaker.get("nxt_agent"))] = (
                speaker.get("global_name"),
                speaker.get("channel"),
            )
    return out


def read_words(path: Path) -> tuple:
    """Return the word file's elements in document order, and an id index."""
    elements, index = [], {}
    for element in ET.parse(path).getroot():
        index[element.get(NITE + "id")] = len(elements)
        elements.append(
            (
                element.tag.split("}")[-1],
                element.get("starttime"),
                element.get("endtime"),
                (element.text or "").strip(),
                element.get("punc"),
            )
        )
    return elements, index


def read_segments(annotations: Path, meeting: str, agent: str) -> Iterator[tuple]:
    """Yield ``(start, end, tokens, punctuation times)`` per transcriber segment.

    The segment elements carry no times of their own; NXT derives them from
    the words, vocal sounds and gaps they dominate, which is why a segment can
    end on a laugh. It takes the first child's start and the *last* child's
    end rather than the extremes: a handful of disfluency markers are stored
    back to front, e.g. ES2011b.C's last word ends at 1548.18 and the marker
    after it at 1547.89, and the segment ends at 1547.89. Segments with no
    ``w`` child are dropped, as the query's ``(exists $w1 w)`` does.
    """
    path = annotations / "segments" / f"{meeting}.{agent}.segments.xml"
    if not path.is_file():
        return
    elements, index = read_words(annotations / "words" / f"{meeting}.{agent}.words.xml")

    for segment in ET.parse(path).getroot():
        children = []
        for child in segment:
            match = HREF.search(child.get("href"))
            first = index[match.group(1)]
            last = index[match.group(2)] if match.group(2) else first
            children.extend(elements[first : last + 1])
        if not any(tag == "w" for tag, *_ in children):
            continue
        starts = [start for _, start, _, _, _ in children if start is not None]
        ends = [end for _, _, end, _, _ in children if end is not None]
        if not starts or not ends:
            continue
        yield (
            float(starts[0]),
            float(ends[-1]),
            [text for _, _, _, text, _ in children if text],
            # TS3011d.B has an untimed full stop. The export writes no field
            # for it, so the segment fails split_transcript's count check and
            # is dropped; dropping the time here is what triggers that.
            [
                start
                for tag, start, _, _, punc in children
                if tag == "w" and punc and start is not None
            ],
        )


def segment_name(start, end) -> str:
    """``get_name``: centiseconds, truncated rather than rounded."""
    return "%07d_%07d" % (int(float(start) * 100), int(float(end) * 100))


def perl_split(pattern: str, text: str) -> list:
    """``split`` as Perl does it: trailing empty fields are dropped."""
    parts = re.split(pattern, text)
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def split_on_comma(text: str, commas: dict, start, end, max_words: int) -> dict:
    """Split ``text`` at the labelled comma nearest its midpoint, recursively.

    Upstream orders the candidate commas by ``cmp``, a string comparison of
    their timestamps, so "9.3" sorts after "10.5"; the loop's ``last`` then
    stops at whichever one happens to equal ``end`` first.
    """
    middle = (float(end) + float(start)) / 2
    key, nearest = "", start
    for candidate in sorted(commas, key=lambda label: commas[label]):
        time = commas[candidate]
        if float(time) == float(start):
            continue
        if float(time) == float(end):
            break
        if abs(middle - float(time)) / 2 < abs(middle - float(nearest)) / 2:
            nearest, key = time, candidate

    out: dict = {}
    if not re.search(r"[,][0-9]+", key):
        out[segment_name(start, end)] = text
        return out

    for index, piece in enumerate(perl_split(re.escape(key) + r"\s+", text)):
        bounds = (start, commas[key]) if index == 0 else (commas[key], end)
        if len(piece.split()) - 1 < max_words:
            out.setdefault(segment_name(*bounds), piece)
        else:
            for name, value in split_on_comma(
                piece, commas, *bounds, max_words
            ).items():
                out.setdefault(name, value)
    return out


def split_transcript(tokens: list, start, end, max_words: int) -> dict:
    """Cut one transcriber segment into ``{name: text}`` at its full stops.

    ``tokens`` is the segment's text followed by one timestamp per punctuation
    mark, which is how the NXT export lays the row out; the two are told apart
    by shape alone, so a spoken "3.5" would be taken for a timestamp and the
    segment dropped on the count check below -- as it is upstream.
    """
    tokens = list(tokens)
    punctuation = [i for i, t in enumerate(tokens) if PUNCTUATION.fullmatch(t)]
    times = [t for t in tokens if TIME_TOKEN.match(t)]
    for i, token in enumerate(tokens):
        if TIME_TOKEN.match(token):
            tokens[i] = None
    while tokens and tokens[-1] is None:
        tokens.pop()

    marks = [tokens[i] for i in punctuation if i < len(tokens)]
    if len(marks) != len(times):
        return {}

    stops = [start]
    stops += [times[i] for i, mark in enumerate(marks) if re.search(r"[.?]", mark)]
    stops.append(end)

    # Commas, exclamation marks and colons are labelled in place -- ",0", ",1"
    # -- so split_on_comma can find them again; full stops are left alone.
    commas: dict = {}
    label_index = 0
    for i, position in enumerate(punctuation):
        label = f"{tokens[position]}{label_index}"
        if re.search(r"[.?].+", label):
            continue
        tokens[position] = label
        commas[label] = times[i]
        label_index += 1

    joined = " ".join("" if t is None else t for t in tokens).upper()
    out: dict = {}
    for index, piece in enumerate(perl_split(r"[.?]", joined)):
        bounds = (
            stops[index] if index < len(stops) else 0,
            stops[index + 1] if index + 1 < len(stops) else 0,
        )
        if len(piece.split()) - 1 < max_words:
            out[segment_name(*bounds)] = piece
        else:
            for name, value in split_on_comma(
                piece, commas, *bounds, max_words
            ).items():
                out.setdefault(name, value)
    return out


def normalise_transcript(text: str) -> str:
    """``normalise_transcripts``: strip the comma labels and the dashes."""
    text = re.sub(r"[.,?!:][0-9]+", "", text)
    # Spurious punctuation inside a word, e.g. "UM,I," -- the whole run goes.
    text = re.sub(r"[A-Z']+,[A-Z']+", " ", text)
    text = text.replace("-", " ")
    # X_M_L spells out as X. M. L.
    text = text.replace("_", ". ")
    text = " ".join(text.split())
    # The perl's two trailing-dash rules sit after its own dash removal and
    # so can never fire; they are left out.
    return text.replace("MM HMM", "MM-HMM").replace("UH HUH", "UH-HUH")


AMI_MIRROR = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
AMI_ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)


class AMIBuilder(OWSMBuilder):
    """Build AMI split caches from the NXT annotations and the headset audio."""

    CORPUS = "ami"

    def prepare_source(self, corpora=None, **_kwargs) -> None:
        """Download AMI's annotations and headset audio (CC BY 4.0, ~100 GB).

        The annotations come first because ``meetings.xml`` says which headset
        channel each speaker wore; only those channels are fetched, which is
        a quarter of what the corpus ships.
        """
        from egs3.owsm_v4.esp2_s2t.dataset import download

        dest = self.source_destination(corpora)
        annotations = dest / ANNOTATIONS_SUBDIR
        download.fetch_and_extract(AMI_ANNOTATIONS_URL, annotations)

        asked = self.corpus_options(corpora).get("meetings")
        listed = {m for names in MEETINGS.values() for m in names}
        if asked:
            listed &= set(asked)
        wanted = sorted(
            {
                (meeting, channel)
                for (meeting, _), (_, channel) in read_meetings(annotations).items()
                if meeting in listed
            }
        )
        download.fetch_many(
            {
                f"{AMI_MIRROR}/{AUDIO_TEMPLATE.format(meeting=m, channel=c)}": dest
                / AUDIO_TEMPLATE.format(meeting=m, channel=c)
                for m, c in wanted
            }
        )
        download.require(dest, f"AMI download left nothing in {dest}")

    def is_valid_source_root(self, candidate: Path) -> bool:
        """Expect the unpacked annotations beside the meeting directories."""
        annotations = candidate / ANNOTATIONS_SUBDIR
        return (
            annotations / "corpusResources" / "meetings.xml"
        ).is_file() and (annotations / "segments").is_dir()

    def utterances(self, annotations: Path, meeting: str, speakers: dict) -> list:
        """Return one meeting's cut utterances as ``(channel, speaker, name, text)``.

        That tuple is the Kaldi utterance id in sort order -- the text file is
        ``sort | uniq``-ed before kaldi_to_whisper.py reads it, and the order
        within a recording breaks ties between equal start times when the
        spans are packed. Two transcriber segments can yield the same cut, so
        the id is deduplicated here too; where the texts differ, the sort
        puts the last one in byte order last and kaldi_to_whisper.py's dict
        keeps that one.
        """
        rows: dict = {}
        for agent in sorted({a for m, a in speakers if m == meeting}):
            speaker, channel = speakers[(meeting, agent)]
            for start, end, tokens, times in read_segments(
                annotations, meeting, agent
            ):
                cut = split_transcript(tokens + times, start, end, MAX_WORDS)
                for name, piece in cut.items():
                    first, last = name.split("_")
                    if int(first) >= int(last):
                        continue
                    text = normalise_transcript(piece)
                    key = (channel, speaker, name)
                    if text and text > rows.get(key, ""):
                        rows[key] = text
        return [key + (rows[key],) for key in sorted(rows)]

    def iter_rows(
        self,
        source_root: Path,
        split: str,
        failures: Iterable | None = None,
        limit: int | None = None,
        read_headers: bool | None = None,
        **_options,
    ) -> Iterator[dict]:
        """Yield cache rows for ``split``, one recording at a time."""
        check_header = READ_HEADERS if read_headers is None else read_headers
        annotations = source_root / ANNOTATIONS_SUBDIR
        speakers = read_meetings(annotations)
        count = 0

        for meeting in MEETINGS[split]:
            by_channel: dict = {}
            for channel, _, name, text in self.utterances(
                annotations, meeting, speakers
            ):
                by_channel.setdefault(channel, []).append((name, text))

            for channel, cuts in by_channel.items():
                path = source_root / AUDIO_TEMPLATE.format(
                    meeting=meeting, channel=channel
                )
                wav_id = f"{self.PREFIX}_{RECORDING_PREFIX}_{meeting}_H0{channel}"
                error = self.audio_error(path, check_header)
                if error is not None:
                    if failures is not None:
                        failures.write(
                            json.dumps({"utt_id": wav_id, "error": error}) + "\n"
                        )
                    continue

                utterances = []
                for name, text in cuts:
                    first, last = (int(x) / 100.0 for x in name.split("_"))
                    utterances.append(
                        Utterance(
                            utt_id=f"{wav_id}_{name}",
                            wav_id=wav_id,
                            wav_path=str(path),
                            start_time=first,
                            end_time=last,
                            lang=f"<{LANG}>",
                            task=f"<{TASK}>",
                            text=text.lower(),
                            asr_text=text.lower(),
                        )
                    )

                for span in generate_long_utterances(utterances):
                    length = round(1000 * span.end_time) - round(
                        1000 * span.start_time
                    )
                    if length <= round(1000 * MIN_DURATION):
                        continue
                    yield {
                        "utt_id": span.utt_id,
                        "wav_path": span.wav_path,
                        "start_time": span.start_time,
                        "end_time": span.end_time,
                        "lang": LANG,
                        "task": TASK,
                        "tgt_lang": "",
                        "text": span.text_with_time,
                        "text_prev": span.prev_text,
                        "text_ctc": span.asr_text,
                    }
                    count += 1
                    if limit is not None and count >= limit:
                        return

    def audio_error(self, path: Path, check_header: bool) -> str | None:
        """Return why ``path`` cannot be used, or None."""
        if not path.is_file():
            return f"no audio at {path}"
        if not check_header:
            return None
        import soundfile as sf

        try:
            info = sf.info(str(path))
        except Exception as exc:  # noqa: BLE001 - recorded, not raised
            return repr(exc)
        if info.samplerate != self.EXPECTED_FS:
            return f"sample rate is {info.samplerate}, expected {self.EXPECTED_FS}"
        if info.channels != 1:
            return f"{info.channels} channels, expected 1"
        return None
