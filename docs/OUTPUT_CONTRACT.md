# Synthetics output contract (version 1)

Oct 21, 2026 (first written Oct 18). This document describes the files that Synthetics hands to the fine-tuning application. Every JSON
file carries `schema_version: "1"`. A change that renames, removes or changes the meaning of a field raises the
version, and new fields are only ever added with defaults, so a reader written for version 1 keeps working.

The Pydantic models in `backend/synthetics/artifacts/` are the source of truth for every field. This document
explains what the fields mean and how the files relate to each other.

## Where the files are

A run lives in `runs/<run_id>/`, or under `SYNTHETICS_RUNS` when that is set.

```
runs/<run_id>/
  config.yaml                  frozen run config (every distribution and setting used)
  speaker_splits.json          NSC speaker -> train / dev / test (fixed by the pool CSV)
  manifest.jsonl               one line per finished meeting (stage 8 passed)
  meetings/<meeting_id>/
    status.json                per-stage state, attempts, times, detail line and metrics
    meeting.json               the sampled room, people, mic, acoustics and device chain (stage 1)
    prompts/<speaker_id>.txt   the persona prompt stage 2 sent for each speaker, verbatim
    script.json                turns with text and text_norm (stage 2)
    script_validation.json     rule checks and judge scores, including persona consistency per speaker (stage 3)
    utterances/<turn_id>.wav   one dry utterance per script turn (stage 4)
    synthesis.json             the voice behind each speaker and how each utterance was made (stage 4)
    voice_validation.json      speaker verification, ASR and MOS checks per utterance, speaker and pair (stage 5)
    timeline.json              when each turn is heard (stage 6)
    tracks/<speaker_id>.wav    dry, time-aligned speech per speaker (stage 6)
    mixture.wav                the single-channel recording (stage 7)
    mixture_mc.wav             the multichannel recording, only for array mics (stage 7)
    render.json                what the acoustic simulation measured and applied (stage 7)
    labels/meeting.rttm        speaker turns (stage 8)
    labels/transcript.json     word-level transcript (stage 8)
    labels/manifest_entry.json this meeting's manifest line (stage 8)
```

The run root also holds `run_report.json` (drops and failures, time per phase, split disjointness; written by
`synthetics report`) and `voice_report.json` (phase C rates; written by `synthetics voice-report`). A run started
from the viewer also has `process.json` (the background process: pid, state, start and end, exit code) and
`process.log` (its output). Those two are for the viewer only and are not part of the output contract.

A meeting is complete when its `status.json` shows `pass` for stage 8. Only complete meetings appear in
`manifest.jsonl`. The manifest's `files` map gives every path relative to the run directory, so a reader never
needs to build paths itself.

## Time and audio

All times are seconds from the first sample of the meeting's audio. Every WAV file of a meeting has the run sample
rate (`sample_rate` in `config.yaml`, 16 kHz by default) and the same number of samples, which is
`timeline.stats.duration` times the sample rate. The files are 16-bit PCM.

- `mixture.wav` is mono. For an array mic it is the mean of the array channels after the device chain, then passed
  through the Opus codec when the meeting's device chain has one. For a single mic it is that mic's channel.
- `mixture_mc.wav` exists only when `meeting.json` has `mic.channels` above 1. Channel `k` is the capsule at
  `mic.position + mic.array_geometry[k]`. It has the device chain up to clipping, but no codec, because the codec
  models how the single-channel stream is sent or stored.
- `tracks/<speaker_id>.wav` holds one speaker's utterances at their timeline positions, dry (no room, noise or
  device), and silence elsewhere. A truncated utterance stops at its cut point with a 10 ms fade. The tracks are
  meant for separation targets and for checking labels against clean speech.

The device chain is applied in this order: band limit, then AGC, then clipping, then the codec. `render.json`
records what was applied, for example the AGC gain range and the share of clipped samples.

## Timeline (`timeline.json`)

`segments` holds one entry per heard turn, sorted by start. Each segment plays its utterance from the start, and
`end - start` is always the audible length. When a turn was cut off by an interruption, `truncated_at` is the
number of seconds into the utterance where it was cut, which equals `end - start`. A speaker's segments never
overlap each other. Segments of different speakers overlap freely, and that overlap is the point of the timeline.

`kind` is the turn's script interaction (`normal`, `interruption`, `backchannel`, `overlap_start`, `side` or
`presentation`). It says what the script asked for, not how the turn was timed, so a `normal` turn may still
start a little before the previous one ends.

`dropped_turns` lists script turns that are not in the audio. Only backchannels are ever dropped. Stage 2 plans its
backchannel count from the sampled meeting length, and when the synthesised meeting comes out shorter, stage 6
leaves some out so that the realised rate stays at the planned one.

`stats` describes the realised turn-taking:

| Field | Meaning |
|---|---|
| `duration` | Length of the meeting audio |
| `speech_time` | Sum of segment lengths (each speaker counted separately); equals the RTTM total |
| `voiced_time` | Time when at least one speaker is active |
| `overlap_time` | Time when two or more speakers are active |
| `overlap_share` | `overlap_time / voiced_time` |
| `overlap_target` | The meeting's sampled target for `overlap_share` |
| `backchannels`, `backchannels_per_min` | Heard backchannels, and their rate over `duration` |
| `speaker_changes` | Transitions between main turns of different speakers |
| `interruptions`, `interruption_share` | Turns cut off by the next speaker, and their share of `speaker_changes` |
| `long_pauses` | Transitions with a long pause (1 to 3 s by default) |

## Voice checks (`voice_validation.json`)

Stage 5 checks every utterance that stage 4 made. These are quality records for the audio, not labels.

- `utterances` has one entry per checked turn: `sv` maps each speaker-verification model (`sv_wavlm`,
  `sv_redimnet`) to the cosine between the utterance and the speaker's reference clip; `asr` holds the ASR
  hypothesis with its WER (or CER for short turns); `mos` is the predicted MOS; `passed` and `reasons` give the
  verdict. `skipped` names checks that were not applied: `sv` for clips under `voice.sv_min_s`, backchannels and
  turns without words, `mos` for clips under `voice.mos_min_s`, and `asr` for turns without words.
  `asr_limited` is true when only the Singlish fallback ASR passed the turn. Phase C no longer runs that fallback,
  so it is always false in current runs.
- `speakers` has one entry per speaker: `consistency` is the mean pairwise cosine between the speaker's own
  utterances of at least `voice.consistency_min_s` (3 s), and `consistency_utts` is how many there were. When
  there are fewer than `voice.consistency_min_utts`, `consistency` is empty and the check is skipped. `ref_mos` is
  the MOS of the reference clip, which the utterance MOS floor is relative to.
- `pairs` has one entry per pair of speakers: `cross_cos` is the mean cosine over every utterance pair across the
  two speakers, and `too_close` is true when either model is above its `thresholds.sv_pair_max`.
- `kept_failed` lists turns that still failed after `retries.tts_attempts`. With `voice.on_exhausted:
  keep_flagged` (the default) the meeting goes on and these turns stay in the audio and the labels. The manifest
  counts them in `voice.kept_failed`. In the M9 run they were 1.3% of utterances.

## Speaker turns (`labels/meeting.rttm`)

The RTTM file has one line per timeline segment, in start order, with millisecond precision:

```
SPEAKER <meeting_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

Speaker labels are the meeting's speaker ids (`S01`, `S02`, and so on, where `S01` is the chair). They are local to
the meeting. The NSC voice behind each label is in the manifest's `speakers` list, taken from
`synthesis.json` (`voices[speaker_id].nsc_speaker`), which is the authoritative voice after any stage 4 reference
swap. `meeting.json` keeps the voice stage 1 first assigned. Segment boundaries are the
utterance WAV boundaries, which keep at most `voice.trim_pad_s` (50 ms) of silence at each end.

## Transcript (`labels/transcript.json`)

`utterances` holds one entry per timeline segment, in the same order as the segments. Each entry has the following
fields:

- `text` uses the Singlish transcription convention from the design document (particles in parentheses, non-speech
  in square brackets, a trailing hyphen for a cut-off).
- `text_norm` is lowercase with markup and punctuation removed. Particles and fillers stay as words. Use it for WER.
- `words` lists every aligned word with absolute `start` and `end` times and the aligner's `score`, spelled as the
  aligner returns it. The aligner drops punctuation and keeps apostrophes, so a word can differ from its `text_norm`
  word only by case and by apostrophes at its edges (`students'` against `students`).
- `truncated` is true when the speaker was cut off. Then `text` and `text_norm` keep only the words heard before
  `truncated_at`, `text` ends with a hyphen, and a word cut in the middle keeps its first part, ending at the cut.
- `align_coverage` is the share of the turn's `text_norm` words that the aligner matched, in order.

The aligner works on each utterance's dry audio, one utterance per request, so a request never holds more than one
turn. The request's language is `zh` for an utterance with Chinese characters and `en` otherwise
(`labels.align_language_cjk` and `labels.align_language`). `aligner` names the service that produced the word
times.

## Manifest (`manifest.jsonl`)

`manifest.jsonl` has one JSON object per line and per complete meeting, sorted by meeting id. The file is rebuilt
from the `labels/manifest_entry.json` files after every stage 8 phase, and with `synthetics manifest -r <run>`.

| Field | Content |
|---|---|
| `run_id`, `meeting_id`, `seed` | The meeting can be generated again from `seed` and `config.yaml` |
| `split`, `room_class`, `n_speakers`, `duration` | For slicing results |
| `sampled` | Every scalar of `meeting.json` with dotted keys (`room.rt60_target`, `mic.type`, `acoustics.snr_db`, `device_chain.codec_kbps`, ...). Per-person data is in `speakers`. `mic.type` is "laptop" (small rooms), "centre" or "chair" (large rooms; "chair" from Oct 22, 2026, a table mic in front of the chair, with `mic.near_speaker` the chair's speaker id) |
| `stages_enabled` | Stage name -> on or off, for ablation runs |
| `position_counts` | Number of people at the table, on wall seats, presenting and remote |
| `speakers` | Per speaker: NSC voice (after any swap), split, position type, wall reason, realised share of speech time, ASR-limited utterance count |
| `judge` | Script rounds, decision, primary and tie-break overall scores and services |
| `voice` | TTS service, utterance, re-synthesis and library-clip counts, reference swaps, voice-check decision, checked and passed counts, ASR-limited total, pairs too close and `kept_failed` (turns kept after failing every attempt) |
| `turn_taking` | The timeline `stats` above, including the raw counts (`voiced_time`, `overlap_time`, `speaker_changes`, `interruptions`, `backchannels`) |
| `acoustic` | Stage 7 measurements (`render.json`, flattened): target and measured RT60, RT60 error, absorption scale, target and realised SNR, channel count, ISM order, ray tracing, impulse responses, noise kinds and sources, event count, `device.*` values the device chain applied, and `remote.<speaker>.*` far-end band and codec |
| `labels` | Stage 8: aligner, utterance and word counts, truncated utterances, minimum and mean alignment coverage, dropped turns |
| `files` | Artifact paths relative to the run directory |

The viewer reads these files directly (see the "Data contract" section of `UI_DESIGN_DOCUMENT.md`).
`synthetics viewer-contract -r <run>` checks every field it relies on, on every meeting of a run, and
`tests/test_viewer_contract.py` runs the same check on a fake run.

Splits are speaker-disjoint. Every speaker in a meeting comes from the meeting's split, and the split of every NSC
speaker is fixed by `data/nsc_part3_reference_speakers.csv` and copied into `speaker_splits.json`.

## Acoustic report (`render.json`)

`render.json` records the stage 7 simulation of a meeting. The room's absorption is scaled until the talkers'
reverberation time matches the sampled target. `rt60_measured` is the median, over the talkers, of the mid-band
T20 (the mean of the 500 Hz and 1 kHz octave bands) measured on each talker's simulated impulse response at the
first mic, and stage 7 fails when it is more than `simulation.rt60_tolerance` (15%) from `rt60_target`.
`snr_db` is speech power over the power of the noise beds (HVAC and projector fan) on the single-channel signal
before the device chain. Keyboard typing and events (door, chair, cup, paper) are set relative to the speech level
and are listed in `noise` with their times and positions.

## Quality gates

Stage 8 fails a meeting, and leaves it out of the manifest, when any of the following checks fails:

1. The RTTM's total speech differs from the sum of the timeline segments by more than the millisecond rounding.
2. An aligned word lies outside its segment by more than `thresholds.align_slack_s` (50 ms).
3. An utterance's alignment coverage is below `thresholds.align_coverage_min` (90%).

Stage 7 also fails a meeting when the measured RT60 is off target, as described above.

## Approximations to know about

- Talker directivity is a cardioid-family pattern (about 8 dB quieter behind the talker). The simulation library
  supports directivity only for early reflections, so in large rooms the ray-traced late reverberation comes from an
  omnidirectional source at the talker's position, scaled by the pattern's average power.
- Remote participants are heard through the room's speakerphone after their own band limit and Opus coding. The echo
  canceller is modelled by its side effects only: local talkers are turned down while a remote participant speaks,
  and the start of each remote turn is attenuated. In small rooms the speakerphone is the laptop's own loudspeaker,
  placed 25 cm from its mic toward the laptop's owner.
- Noise comes from the noise library, `$SOUND_LIBRARY_DIR/noise/index.yaml` (default
  `/home/user/code/datasets/sound-library/noise`), built by `data/sound_library/build_noise_library.py`. Kinds
  without library clips (projector fan, chair, paper) are procedural (filtered noise and tones). `render.json` names
  the source of every noise. The room-tone beds are DEMAND office recordings (CC BY-SA 3.0); they were recorded in a
  real room, so they are mixed per capsule without a simulated impulse response. Keyboard, door and cup sounds come
  from ESC-50, which is licensed CC BY-NC 3.0 as a whole (non-commercial use only), with each clip's Freesound
  source and original licence in the index. Every clip passed a check for pitched, speech-like frames, so the noise
  holds no unlabelled speech.
- The planned overlaps alone fall a little short of the 10 to 25% target. With the default turn-taking settings
  (since Oct 21, 2026, 50% of transitions overlapping by 0.3 to 2.0 s) they give about 12 to 15% of speech time.
  Runs made before then used 30% and 0.1 to 1.0 s, which gave 5 to 8%. Stage 6 meets the target by lengthening the
  overlaps (up to three times, and never more than half of either turn) and then by letting more transitions overlap.
  `status.json` records both as stage 6 metrics (`lead_scale`, `converted_transitions`). A meeting with few
  speaker changes (for example two people, one of them presenting) can run out of transitions to overlap; it then
  stays under its target, still inside the configured range, and stage 6 says "out of reach" on its status line.

## Changing sampled acoustics (run revision 2)

Changing a render setting (`simulation` in `config.yaml`) re-runs only stages 7 and 8. Changing a sampled acoustic
distribution (`acoustics` in `config.yaml`, for example the SNR range) re-runs stages 1, 7 and 8, and leaves the
scripts and the speech alone. This holds for runs with `revision: 2` in `config.yaml`, which is the default for new
runs since Oct 21, 2026:

1. Stage 1 draws the acoustic values (`room.walls`, `room.floor`, `room.ceiling`, `room.rt60_target`, `acoustics`
   and `device_chain`; `ACOUSTIC_FIELDS` in `artifacts/meeting.py`) from a random stream of their own, so a change
   to them never moves the other draws.
2. Stages 2 to 6 hash `meeting.json` without those fields in their cache keys.

Runs made before then (`m3-10`, `m5-exit`, `m9-e2e`) have no `revision` in their frozen config, so they are
revision 1: their meetings keep the draws they were made with and their cache keys do not change, and a change to
their acoustics still re-runs every stage. The same seed draws different meetings under the two revisions.
