# Running this recipe on a machine that has nothing

A licence-clean OWSM v4/v5 trains from raw corpora: there is no published dump
to copy and no `egs2/owsm_v4/s2t1/local/` preparation to run. Everything below
is what a fresh machine needs, in the order it needs it.

## 1. A Hugging Face token, with `espnet` org access

Three repositories the recipe reads are **private**:

| Repository | Read by |
|---|---|
| `espnet/common_voice_13_0_mirror` | `commonvoice`, and `covost2` for its audio |
| `espnet/yodas2-clean-metadata` | `yodas2_cleaned` |
| `espnet/yodas3-clean-metadata` | `yodas3_cleaned` (owsm_v5 only) |

A token alone is not enough: the account must be a member of the `espnet`
organisation. Export `HF_TOKEN`, or run `huggingface-cli login`. Set `HF_HOME`
to somewhere with room — the metadata alone is 11 GB for YODAS2 and 23 GB for
YODAS v3.

The Common Voice mirror exists because Mozilla emptied
`mozilla-foundation/common_voice_13_0` in October 2025, when the corpus moved
to the Mozilla Data Collective. That repository still resolves and still
serves a README, so a fetch from it *succeeds* and yields no corpus.

## 2. Environment variables

Eleven corpus roots plus the cache. `path.sh` only warns about two of them.

    AISHELL  AMI  COMMONVOICE  COVOST2  FLEURS  LIBRISPEECH  MLS  VCTK
    VOXPOPULI  YODAS2_CLEANED  YODAS3_CLEANED  OWSM_CACHE

Each is read as `${oc.env:NAME,${recipe_dir}/download/<corpus>}`, so **unset
means "download it here", not "fail"**. A typo therefore starts a
multi-terabyte download instead of using a corpus the machine already holds.
Set `OWSM_CACHE` to an absolute path on a large filesystem.

Two further variables are optional. `MLS_LONG_AUDIO` and `LIBRISPEECH_MP3`
name the second tree each of those corpora needs; leave them unset and both
are created beside the corpus by `prepare_source`.

## 3. Do not `source ./path.sh`

It sources `../../../tools/activate_python.sh`, which is not in a worktree
that has not been through espnet's `tools/` Makefile. The `PYTHONPATH` export
happens first, so the script limps rather than dies, but it activates no
Python environment. Instead:

    W=<repo root>
    cd $W/egs3/owsm_v4/esp2_s2t
    export PYTHONPATH=$W:$(pwd)

`recipe_dir` is `.`, so `exp_dir`, `data_dir`, `stats_dir` and the tokenizer
path all resolve against the working directory. Run from the recipe.

## 4. The stages, in order

    python run.py --stages create_dataset       --training_config conf/<config>.yaml
    python run.py --stages train_tokenizer,collect_stats --training_config conf/<config>.yaml
    python run.py --stages train                --training_config conf/<config>.yaml

`train` runs `train_tokenizer` by itself when the tokenizer is missing. It
does **not** run `collect_stats`, and the model normalises with `global_mvn`
from `${stats_dir}/train/feats_stats.npz`, so that stage is not optional.

## 5. `create_dataset` is the long pole, and belongs on a CPU node

Rough sizes, from each builder's own docstring:

| Corpus | Download |
|---|---|
| MLS | ~2.4 TB for all eight languages |
| Common Voice | 0.627 TB |
| AMI | ~100 GB, headset channels only |
| LibriSpeech | ~140 GiB, flac splits plus original-mp3 |
| VoxPopuli | ~70 GB |
| AISHELL-1 | ~15 GB |
| VCTK | 11.7 GB |
| CoVoST 2 | ~250 MB, manifests only; audio is Common Voice's |
| FLEURS | via `datasets.load_dataset` |

MLS then rebuilds the recordings OWSM cuts its spans from: 203,626 of them,
180,605 in English train alone. The builder does this — there is no external
script — but it is why `parallel.n_workers` is 128 in the shipped configs.
That value governs every `run_parallel` call site and overrides the
per-corpus `workers` in the sub-dataset configs, so lowering it to 1 makes
the CommonVoice duration probe and the MLS rebuild serial.

`parallel.n_workers` has no effect on `--stages train`; the training loaders
use `dataloader.train.num_workers` instead.

## 6. The tokenizer holds the whole mixture in memory

`tokenizer.sample_size: null` means every line goes into a Python list *and*
a single joined string. At 27M rows that is survivable; with real cleaned
YODAS2 in the mixture it is roughly 50M. Either set `tokenizer.sample_size`
to about 10M, or skip the stage: drop the published OWSM v4 BPE at
`${data_dir}/bpe_50000` (it needs `bpe.model` and `tokens.txt`) and
`_has_tokenizer()` will skip it. All 1,681 symbols this recipe emits are
already in that vocabulary, so the result stays comparable with the published
models.

## 7. Cleaned YODAS2 is its own project

`yodas2_cleaned` fetches metadata and then transcodes the audio it references:
195,487 hours, order a terabyte of Opus and thousands of CPU-hours. It has
never been built at full scale. Budget it separately from everything above.

## 8. Watch Hugging Face egress

Pulling the mirrors onto another cluster counts against the account's monthly
egress, and the Hub warns before it throttles:

> You have used a large amount of egress over the last 30 days. To protect our
> platform health, once you reach your account's max quota, your downloads
> will be throttled or blocked.

Download each corpus once, to a shared filesystem, rather than per user.

## 9. `batch_size` is per rank

Unlike espnet2, where it is the global total. The shipped configs are
20 per rank x 2 devices x `accumulate_grad_batches` 8 = a global batch of 320.
To change the GPU count, scale `accumulate_grad_batches` and leave
`batch_size` alone, or the warmup schedule stretches against the data.

`multiprocessing_context: spawn` on both loaders is not optional: forked
workers die with "CUDA error: initialization error". espnet3 forces
`reload_dataloaders_every_n_epochs = 1`, so the loaders are rebuilt at every
epoch boundary and an import that survives step 0 can still fail at epoch 2.
