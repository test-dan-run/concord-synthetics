# Synthetics output contract (version 1)

Oct 18, 2026. This document describes the files that Synthetics hands to the fine-tuning application. Every JSON
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
    meeting.json               the sampled room, people, mic, acoustics and device chain (stage 1)
    script.json                turns with text and text_norm (stage 2)
    timeline.json              when each turn is heard (stage 6)
    tracks/<speaker_id>.wav    dry, time-aligned speech per speaker (stage 6)
    mixture.wav                the single-channel recording (stage 7)
    mixture_mc.wav             the multichannel recording, only for array mics (stage 7)
    render.json                what the acoustic simulation measured and applied (stage 7)
    labels/meeting.rttm        speaker turns (stage 8)
    labels/transcript.json     word-level transcript (stage 8)
    labels/manifest_entry.json this meeting's manifest line (stage 8)
```

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

## Speaker turns (`labels/meeting.rttm`)

The RTTM file has one line per timeline segment, in start order, with millisecond precision:

```
SPEAKER <meeting_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

Speaker labels are the meeting's speaker ids (`S01`, `S02`, and so on, where `S01` is the chair). They are local to
the meeting. The NSC voice behind each label is in the manifest's `speakers` list. Segment boundaries are the
utterance WAV boundaries, which keep at most `voice.trim_pad_s` (50 ms) of silence at each end.

## Transcript (`labels/transcript.json`)

`utterances` holds one entry per timeline segment, in the same order as the segments. Each entry has the following
fields:

- `text` uses the Singlish transcription convention from the design document (particles in parentheses, non-speech
  in square brackets, a trailing hyphen for a cut-off).
- `text_norm` is lowercase with markup and punctuation removed. Particles and fillers stay as words. Use it for WER.
- `words` lists every aligned word with absolute `start` and `end` times and the aligner's `score`. The words are the
  `text_norm` words.
- `truncated` is true when the speaker was cut off. Then `text` and `text_norm` keep only the words heard before
  `truncated_at`, `text` ends with a hyphen, and a word cut in the middle keeps its first part, ending at the cut.
- `align_coverage` is the share of the turn's `text_norm` words that the aligner matched, in order.

The aligner works on each utterance's dry audio, one utterance per request, so a request never holds more than one
turn. `aligner` names the service that produced the word times.

## Manifest (`manifest.jsonl`)

`manifest.jsonl` has one JSON object per line and per complete meeting, sorted by meeting id. The file is rebuilt
from the `labels/manifest_entry.json` files after every stage 8 phase, and with `synthetics manifest -r <run>`.

| Field | Content |
|---|---|
| `run_id`, `meeting_id`, `seed` | The meeting can be generated again from `seed` and `config.yaml` |
| `split`, `room_class`, `n_speakers`, `duration` | For slicing results |
| `sampled` | Every scalar of `meeting.json` with dotted keys (`room.rt60_target`, `mic.type`, `acoustics.snr_db`, `device_chain.codec_kbps`, ...). Per-person data is in `speakers` |
| `stages_enabled` | Stage name -> on or off, for ablation runs |
| `position_counts` | Number of people at the table, on wall seats, presenting and remote |
| `speakers` | Per speaker: NSC voice (after any swap), split, position type, wall reason, realised share of speech time, ASR-limited utterance count |
| `judge` | Script rounds, decision, primary and tie-break overall scores and services |
| `voice` | TTS service, utterance and re-synthesis counts, reference swaps, voice-check decision and ASR-limited total |
| `turn_taking` | The timeline `stats` above |
| `acoustic` | Stage 7 measurements: target and measured RT60, SNR, channel count, noise kinds and sources, device chain values, far-end settings of remote speakers |
| `labels` | Aligner, word and utterance counts, truncated utterances, alignment coverage, dropped turns |
| `files` | Artifact paths relative to the run directory |

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
- Noise beds, keyboard typing and events are procedural (filtered noise and tones) until real recordings are added to
  `datasets/sound-library/noise/index.yaml`. `render.json` names the source of every noise.
- With the default turn-taking settings (30% of transitions overlapping by 0.1 to 1.0 s), the realised overlap is
  about 5 to 8% of speech time, which is under the 10 to 25% target. Stage 6 meets the target by lengthening the
  overlaps (up to three times, and never more than half of either turn) and then by letting more transitions overlap.
  `status.json` records both as stage 6 metrics (`lead_scale`, `converted_transitions`).

## Open issue: changing sampled acoustics re-runs every stage

Changing a render setting (`simulation` in `config.yaml`) re-runs only stages 7 and 8, as intended. Changing a sampled
acoustic distribution (`acoustics` in `config.yaml`, for example the SNR range) currently re-runs every stage. Stage 1
samples those values into `meeting.json`, and the runner's cache key hashes the whole of `meeting.json` for stages 2
to 6. Two upstream changes would fix it:

1. In stage 1, draw the acoustic values (`room.walls`, `room.floor`, `room.ceiling`, `room.rt60_target`, `acoustics`,
   `device_chain`) from their own random stream, so a change to them never moves the other draws.
2. In the runner, hash `meeting.json` without those fields for stages 2 to 6.

`tests/test_post.py::test_changing_sampled_acoustics_reruns_only_stages_7_and_8` is a strict expected failure until
both changes are made.
