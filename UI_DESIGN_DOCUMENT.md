# Room Config Viewer — UI Design

Oct 18, 2026 · @Dan

## Purpose and goals

The Room Config Viewer is the Room viewer section of the Synthetics product: a web UI for inspecting what the synthetic Singapore meeting pipeline generated, from the sampled rooms, positions, mics and personas of each meeting to whether the whole run matches its configured distributions.

Two product decisions shape everything below:

- **Part of Synthetics, outside Concord.** The viewer ships in the Synthetics app under its own name and icon, built on the AeroAudio design system so it looks and behaves like the other workspace tools.
- **Read-only room viewer.** The room viewer reads pipeline artifacts from `runs/<run_id>/` and never writes back. It has no resample, regenerate or edit actions; its one action is exporting a manifest slice.
- **A separate run builder (added Oct 21, 2026, M14).** Creating and running a dataset lives in its own sidebar group, Generate, so the room viewer's screens stay read-only. The builder only creates new runs and starts or stops them; it never edits a frozen config or a stage's output. `synthetics viewer --read-only` leaves it out.

Goals:

- Let a pipeline engineer check any single meeting's geometry, positions and persona setup in under a minute.
- Make sampler drift visible: room-class mix, presentation and wall-seating rates, hybrid rate, acoustics, mic distances and turn-taking, each against `config.yaml`.
- Show each persona's saved prompt beside the script turns it produced, so script quality problems trace back to the prompt.

Out of scope: editing an existing run's config, re-running single stages or meetings, audio playback, label review, and anything the fine-tuning application owns (training, evaluation).

## Users and key tasks

Two groups use the viewer: the pipeline team, who build and tune the generator, and the fine-tuning team, who consume its output and want to understand what is in a run.

| Task | Who | Where |
| --- | --- | --- |
| Find meetings by room class, speaker count, split, hybrid, presenters or wall seats | Both | Meeting browser |
| Spot failed or still-running meetings | Pipeline | Meeting browser tiles and card status |
| Check that seats, table, wall chairs, presenter spots and mic placement are physically plausible | Pipeline | Meeting detail floor plan and elevation |
| See each speaker's position, distance and facing relative to the mic | Both | Meeting detail floor plan, speaker table |
| Read the full sampled config (room, positions, surfaces, noise, device chain) | Both | Meeting detail config panels |
| Inspect a persona's prompt and the turns it produced | Pipeline | Meeting detail persona panel |
| Compare run-level distributions, including presentation and wall-seating rates, with configured targets | Both | Sampler distributions |

## Information architecture and navigation

The viewer has three screens inside the AeroAudio app shell of Synthetics: a sidebar headed by the Synthetics name and icon, for navigation and run selection, and a main column of stacked panels.

&#91;embedded content: screens and navigation · 3 screens\]

The sidebar links to Meetings and Distributions, then under Generate to New dataset and Pipeline (screens 4 and 5, left out in read-only mode), and holds the active-run select plus a jobs card for any batch in progress. Changing the run reloads every screen for that run. Meeting cards open the detail screen; crumbs lead back, and Previous and Next step through meetings in the browser's current filter and sort order.

## Screen 1: Meeting browser

The browser is the entry point: a run's meetings as a grid of cards, each with a to-scale mini floor plan, filtered from one panel above the grid.

Layout, top to bottom:

1. **Page header.** Crumbs, title, run summary line (meeting count, seed, config hash), and one action, Export manifest slice (primary); being read-only, the screen has nothing else to trigger.
2. **Metric tiles.** Meetings (done count), Small rooms and Large rooms (count, share, target), With presenters (share of meetings), Wall seats (share of large rooms, with the overflow count), Hybrid share, and Needs attention (failed plus running). Room-class tiles carry their category swatch.
3. **Filters panel.** One row that wraps: Room class toggle (All, Small, Large, each with a count), Split toggle (All, train, dev, test), Min speakers slider (2–20), and "Only meetings with" checkboxes for remote participants, presenters and wall seats. Filters apply immediately; there is no Apply button. A marker legend runs along the bottom of the panel.
4. **Result bar.** "N of M shown" count and a Sort select (meeting ID, speakers, RT60, SNR).
5. **Card grid.** Auto-fill columns, minimum 300 px. Each card links to the meeting detail.
6. **Load more.** Pages in 12 cards at a time.

Each card shows:

- **Mini floor plan** on a 1 m grid, scaled to fit 330 × 184 px: room outline, table (rect, oval or round), table seats as hollow circles in the room-class colour, wall seats as solid circles, presenters as hollow squares, the chair as a filled white seat, the mic as a diamond, and the screen as a bar on the front wall when the meeting has presenters. Room dimensions sit bottom-left; a "+N remote" chip top-right for hybrid meetings.
- **Meeting ID** and a room-class chip.
- **Speakers, RT60, SNR** as three labelled numbers, then a positions line: how many sit at the table, at the wall (overflow or by choice), present, or join remotely.
- **Tags** for mic type, table shape and split, plus a status dot and word (done, running, failed).

States: an empty filter result shows the system's dashed empty state ("No meetings match these filters") with a Clear filters button; loading shows the spinner in the result bar; a run that failed to load shows an error note with the path that was tried.

## Screen 2: Meeting layout detail

The detail screen shows one meeting's sampled setup to scale, and everything on it is keyed to one selected speaker: clicking a seat, a table row or the persona dropdown changes the selection everywhere at once. The default selection is the speaker with the second-highest turn share, so the chair does not always open first.

Layout, top to bottom:

1. **Page header.** Crumbs back to Meetings, the meeting ID as the title, chips for room class, speaker counts by position (table, wall, presenter, remote), hybrid, presentation and split, the seed as a tag, and actions View meeting.json, Previous and Next.
2. **Pipeline status.** Eight stage tiles, each with a status dot, stage number and name, status word and one detail (resamples, turn count, judge retries, re-synthesised utterances, overlap share).
3. **Floor plan and side panels**, side by side, wrapping on narrow screens.
4. **Persona panel**, full width.
5. **Speakers and positions table**, full width.

### Floor plan

A top-down plan drawn at 90 px per metre, scrolling inside its panel if wider than the column.

- **Walls** carry their material: glass dashed, plasterboard solid, each labelled with its side (N, E, S, W). Room length and width, RT60, floor, ceiling and height are printed around the outline.
- **Table** as its true shape and size; **speakerphone** on the table for hybrid meetings, labelled with the remote speakers it carries.
- **Mic** as a diamond with its type, channel count and height.
- **Seats** are buttons labelled with the speaker number; diameter scales with turn share; the chair has a heavier ring; the selected seat is filled. Free table seats are dashed outlines.
- **Layers** toggle group: Facing (a tick showing each speaker's direction), All mic paths (a faint line from every seat to the mic) and Grid (1 m squares). The selected speaker's mouth-to-mic path is always shown as a dashed line with its 3D distance.

Off-table positions follow the backend's three position types:

- **Front wall and screen.** The front wall is marked, with the screen as a bar on it. Room orientation is fixed so the front wall is always on the left.
- **Presenters** are rounded-square buttons standing 0.5–1.5 m from the front wall, with a dashed 1 m circle for the drift allowed between turns. Their facing tick points at the audience; the facts note that they face the screen on 30% of turns. If the script returns a presenter to a seat, that free table seat is labelled ("S03 returns here").
- **Wall seats** (large rooms only) are seat buttons on a strip of wall chairs 0.3–0.6 m from a side or back wall. The facts say whether they are overflow (the table was full) or by choice (the table had free seats).
- Every position, on or off the table, gets a mouth-to-mic path and 3D distance, so the farther, more reverberant pickup of wall seats and presenters is visible.

### Side elevation

Below the plan, a side view at the same scale shows floor, ceiling, table height, mic height and every mouth height. The selected speaker's path to the mic is drawn with its mouth height labelled. North and south seats overlap in this view, and a note says so. Presenters are drawn standing, with a line from the floor to the mouth at 1.5–1.75 m; wall seats sit at the back wall.

### Selected speaker and config

The right column holds a selected-speaker card (role, position, L1, code-switching, turn share and utterance count, mouth height, 3D mic distance, facing, NSC reference, both speaker-verification scores, ASR-limited count) and seven fact panels from `meeting.json`: Room (including which wall is the front), Table seats (seats used of total), Off-table positions (wall seats and why, presenter spot, mouth height, drift, presentation segment times, where the presenter goes after), Microphone, Surfaces, Noise, Device chain.

### Persona panel

The persona panel explains why a speaker sounds the way they do, by putting the generation prompt next to its output.

- **Header:** persona ID and role, generator model, prompt template version, the judge's persona-consistency score, and a persona select.
- **Persona prompt:** the exact prompt stage 2 sent for this speaker, read from its saved prompt file and shown as is, never rebuilt from a template, in a scrolling monospace block, with a Copy prompt button. Its first line names the speaker's position (table seat, wall seat, presenter or remote), and presenters' and wall-seated speakers' settings are spelled out. It covers the meeting brief, profile (seniority, verbosity, L1, code-switching level, turn-share target), stance in this meeting and output rules.
- **Sample from script.json:** a short excerpt of turns around this persona with time, speaker, interaction type and text. The persona's own turns are highlighted; overlap starts carry the overlap hatch, and presenters' monologue turns are typed presentation. A text / text\_norm toggle switches between the readable transcript and the normalised form used for WER.
- **Footer:** turns shown, how many are this persona's, the persona's turn target, and Open full script.

### Speakers and positions table

One row per speaker, including remote ones: speaker (a button that selects), position (table seat, wall seat, presenter or remote), persona, L1 and code-switching, turn-share bar and percentage, mic distance, facing and NSC reference. The selected row is highlighted.

## Screen 3: Sampler distributions

This screen answers one question: did the run produce what `config.yaml` asked for? Every chart shows the actual distribution for the run with the configured range shaded behind it.

**Target tiles** (one wrapping row):

- **Room class mix:** two stacked bars, actual versus target (30% small, 70% large by default).
- **Presentation segment:** actual share of meetings with presenters per class, with the gap from target (30% small, 40% large) in percentage points, plus how many presentations have two presenters.
- **Hybrid meetings:** actual share per class with the gap from target in percentage points, using the system's delta style. A gap above 2 points is shown as a regression.
- **Wall seating:** share of large rooms with wall seats, split into overflow and by choice (target p = 0.15), and the most people at the wall in one meeting.
- **Needed a room resample:** share of meetings where the table, wall chairs and presenter space did not fit on the first draw, with the maximum resample count.
- **Splits:** train, dev and test meeting counts, plus table-shape counts.

**Charts:**

| Group | Chart | Bins | Configured range shown |
| --- | --- | --- | --- |
| Speakers | Speakers per meeting, all positions, both classes | 1 speaker, 2–20 | Bars coloured by class (2–6 small, 7–20 large) |
| Rooms and acoustics | RT60, small and large rooms | 0.05 s, 0.2–1.0 s | 0.3–0.6 s small, 0.4–0.9 s large |
| Rooms and acoustics | SNR at mic, small and large rooms | 2 dB, 0–32 dB | 10–30 dB small, 5–25 dB large |
| Rooms and acoustics | Mouth-to-mic distance, table seats, laptop and centre mic | 0.25 m | 0.3–0.5 m nearest seat for laptop; centre offset for array |
| Positions | Wall-seated speakers per meeting, large rooms | 1 speaker, 1–8 | None (overflow depends on table capacity) |
| Positions | Mouth-to-mic distance, wall seats, centre mic | 0.25 m, 0–6 m | None |
| Positions | Mouth-to-mic distance, presenters, small and large rooms | 0.25 m, 0–6 m | None |
| Turn-taking | Overlap share of speech time, small and large rooms | 2%, 0–36% | 10–20% small, 15–25% large |
| Turn-taking | Backchannels per minute, all meetings | 0.25 per min, 0–5 | 1–3 per min |
| Turn-taking | Interruptions as % of turn changes, all meetings | 1%, 0–16% | 5–10% |

Charts sit under three group headings: Rooms and acoustics (stages 1 and 7), Positions (stage 1) and Turn-taking (stage 6). Per-class charts are small multiples on the same axis range, so the two classes compare directly; all-meeting charts use `color-primary` as the single series. Each chart has a one-line note under it. Hovering a bar dims its neighbours and shows a tooltip with the bin range and count.

## Screen 4: New dataset (run builder)

Added Oct 21, 2026 (M14). One page at `/new`: a form on the left and a sticky preview on the right, stacked below `xl`.

1. **Run.** Start from the defaults or from an existing run, run id (suggested as `d-<date>`, checked against the runs on disk), meetings, seed, and backends (real models on the GPU, or fake with no GPU). Starting from a run copies its frozen config: the form fields fill in, and every other setting that differs from the defaults becomes a line of the YAML overlay. The new run always takes the current revision.
2. **Curated settings,** grouped as rooms and speakers, turn-taking, seating and participants, acoustics and device, splits, and stage switches (3, 5 and 7). Ranges are two numbers, probabilities a slider with a number in %, weights one number per category with its normalised share. A field that differs from the default says "changed". Every setting, the run fields and the YAML overlay included, has an info button beside its label. On hover, keyboard focus or a tap it explains what the setting is and why it matters for the training data (the text lives with the field in `lib/draft.ts`). Escape closes it, and it stays inside the screen at phone width.
3. **YAML overlay** for any other setting, as in a `--config` file; dotted keys work too, and the overlay wins over the form.
4. **Preview.** Stage 1 runs on the server in memory for the draft, a short pause after each change, so it shows the meetings the run will sample (up to the first 500): class and split counts, audio hours, hybrid, presentation and wall seat shares, mini plans of the first six meetings, histograms of speakers, length, RT60 and SNR, the sampler checks, and any meeting that cannot be sampled. Validation errors from the backend appear under their field.
5. **Create.** Says whether the run will lease GPU services and, for real backends, a rough time scaled from `m9-e2e`. Creating writes the frozen config and goes to the Pipeline screen; it does not start the run.

## Screen 5: Pipeline

`/runs/<run>/pipeline`. Start (or Resume once any stage has run) and Stop in the header, a Process panel (state, start and end, exit code, error), a Stages table (meetings per state for each stage, from `status.json`), and the tail of `process.log`, refreshed every 3 seconds while running. Starting a run on real backends opens a dialog listing the services each GPU phase leases. Stop sends an interrupt, as Ctrl-C would, so the lease releases its services, and Resume continues from the status files. Only one run can run at a time.

## Visual design

The viewer uses AeroAudio throughout: dark slate only, Inter for text, JetBrains Mono with tabular figures for every number, ID and unit, flat surfaces with 1 px borders, and the standard Sidebar, PageHeader, Panel, MetricTile, Toggle, Chip, Tag, Fact and SortableTable components. It adds two things to the system: a categorical palette and the Synthetics icon, both below.

Colour carries meaning only where the system assigns it:

| Meaning | Token | Used for |
| --- | --- | --- |
| Small room category | `color-cat-1` (blue) | Seats on small-room plans, tile swatches, chart bars |
| Large room category | `color-cat-2` (orange) | Seats on large-room plans, tile swatches, chart bars |
| Single series | `color-primary` | All-meeting charts such as backchannels and interruptions |
| Selection | `color-focus` (sky) | Selected seat, mouth-to-mic path, distance label, elevation highlight |
| Done | `color-good` | Meeting and stage status dots |
| Running | `color-primary` | Running status, job progress |
| Failed or regression | `color-bad` | Failed status, hybrid gap over 2 points |
| Overlap | `color-overlap` hatch | Overlap-start turns in the persona excerpt |

### Categorical palette

AeroAudio gains a categorical palette, `color-cat-1` to `color-cat-6`, for any chart or drawing whose series are categories rather than status. Assign slots in this fixed order and never cycle them; a seventh category folds into Other or a small multiple.

| Token | Hue | Value |
| --- | --- | --- |
| `color-cat-1` | Blue | `#3987e5` |
| `color-cat-2` | Orange | `#d95926` |
| `color-cat-3` | Aqua | `#199e70` |
| `color-cat-4` | Yellow | `#c98500` |
| `color-cat-5` | Magenta | `#d55181` |
| `color-cat-6` | Violet | `#9085e9` |

On `color-panel` all six sit inside the dark-mode lightness band (L 0.48–0.67), pass 3:1 contrast, and keep adjacent pairs apart for colour-blind readers (worst ΔE 8.4) and full-colour readers (worst ΔE 19.3). The first three also separate in any pairing, so scatter plots and small multiples cap at three categories. Aqua sits near `color-good`, so categories never stand in for status, and status always carries its word.

Individual speakers are still not coloured: meetings reach 20 speakers, beyond any palette, so they are identified by number and highlighted by selection.

Plan drawing conventions:

- Floor `color-canvas`, 1 m grid in `color-track`, table `color-active` with a `color-line` edge.
- Glass walls dashed `color-muted`; plasterboard solid `color-subtle`. Pattern, not just tone, tells them apart.
- The front wall is always drawn on the left, with the screen as a `color-focus` bar on it.
- Position types differ by shape, never by colour alone: table and wall seats are circles, presenters are rounded squares, free table seats are dashed circles. Wall seats sit on a `color-track` strip of wall chairs; on browser thumbnails they are solid circles.
- Seats are outlined in `color-fg`; the chair's outline is heavier; size scales with the square root of turn share.
- A presenter's drift area is a dashed `color-subtle` circle of 1 m; in the elevation a presenter is a line from the floor to the mouth.
- Mic is a `color-fg` diamond; facing ticks are `color-subtle`.

### Synthetics icon

Synthetics has its own mark, separate from the Concord icon set, so the product reads as its own app. It follows AeroAudio's icon rules: a 24-unit grid, 1.8 stroke, round caps and joins, no fill, drawn in `currentColor`.

&#91;embedded content: Synthetics icon · grid, sizes and meaning\]

In the sidebar it sits at 22 px in `color-primary` beside the Synthetics name, as the AeroAudio logo slot specifies. Other icons in the viewer stay as text labels (Previous, Next) until a Synthetics icon set exists.

## Data contract

The viewer reads the pipeline's storage layout directly and computes everything else in the browser; it needs no new backend fields except where noted.

| Artifact | Read by | Fields used |
| --- | --- | --- |
| `manifest.jsonl` | Browser, Distributions | Meeting ID, room class, speaker count, wall-seated and presenter counts, per-speaker position type, mic type, hybrid, RT60, SNR, split, tags |
| `config.yaml` | Distributions | Room-class ratio, per-class ranges for speakers, RT60 and SNR, hybrid probabilities |
| `meeting.json` | Browser cards, Detail | Room dimensions, front wall and screen, table shape and size, seat positions and mouth heights, each speaker's position type (table, wall, presenter), wall-seat placement and reason (overflow or by choice), presenter spot, drift and return seat, presentation segment times, mic type and position, surfaces, noise, device chain, persona fields per seat, turn shares |
| `status.json` | Browser, Detail | Per-stage status, timings, retry and resample counts |
| `script.json` | Detail persona panel | Turns with speaker, interaction type, `text`, start time |
| `script_validation.json` | Detail persona panel | Judge persona-consistency score |
| `voice_validation.json` | Detail speaker card | Both speaker-verification scores, ASR-limited flags |
| `timeline.json` | Detail persona panel | Turn start times when the script has none |
| `speaker_splits.json` | Detail | NSC reference speaker per persona |
| `prompts/<speaker_id>.txt` | Detail persona panel | The rendered persona prompt stage 2 sent for each speaker, shown verbatim |
| `process.json`, `process.log` | Pipeline, jobs card | State of a run started from the viewer, and its log (M14) |

Computed in the UI: mouth-to-mic distance in 3D, facing angle relative to the table centre, table edge per person, room volume, histogram bins, and `text_norm` when it is not already stored.

New fields needed from the pipeline (all in place since M10, Oct 21, 2026):

- The **rendered persona prompt** per speaker, written by stage 2 to `prompts/<speaker_id>.txt` alongside `script.json`.
- **Turn-taking statistics** per meeting in `manifest.jsonl` (`turn_taking`), computed by stage 6 from the timeline: overlap share of speech time, backchannels per minute, interruptions as a share of turn changes, plus the raw counts.
- A **fit resample count** in `meeting.json` (`fit_resamples`), also in the manifest's `sampled` and the stage 1 metrics in `status.json`.
- **Speaker facing per turn: not in v1.** Decided Oct 21, 2026: the plan shows the one sampled direction per seat (`people[].facing_deg`). Stage 7 does turn talkers between turns, but it does not write those directions out; per-turn facing waits until a later viewer version asks for it.

Where the other fields live: mic type, table shape and hybrid are `sampled["mic.type"]`, `sampled["table.shape"]` and `sampled["dynamics.hybrid"]` in the manifest; wall-seated and presenter counts are `position_counts`; sampled RT60 and SNR are `sampled["acoustics.rt60"]` and `sampled["acoustics.snr_db"]`, and the measured values are in `acoustic`. The NSC reference speaker per persona is `speakers[].nsc_speaker` in the manifest (or `synthesis.json`), which reflects any stage 4 voice swap; `meeting.json` keeps the first assignment. `asr_limited` is always false in current runs, because phase C no longer runs the Singlish fallback ASR, so the speaker card shows it as "n/a" when no utterance is flagged. `synthetics viewer-contract -r <run>` checks every row of this table on a run.

## Accessibility and responsive behaviour

The viewer follows AeroAudio's accessibility rules, with a few additions for the drawings.

- **Keyboard:** every seat on the plan is a real button reachable by Tab, with an accessible name ("Select S02, Engineering lead") and `aria-pressed` for the selected seat. Layer toggles, class and split filters use `aria-pressed`. The global focus ring is never removed.
- **Colour is never the only signal:** room-class chips carry the class name, status dots sit next to a status word, wall materials differ in pattern and are labelled, and the selected path carries its distance as text.
- **Plans and charts:** each plan has a text equivalent in the speaker table (distance and facing per speaker). Each histogram needs a table view toggle before release; the mock-ups only have hover tooltips.
- **Semantics:** panels are sections labelled by their h2; tables have captions and `scope="col"` headers; navigation is `nav aria-label="Primary"`.
- **Contrast:** text uses `color-fg` or `color-muted` only; `color-subtle` is limited to facing ticks and the plasterboard line, never to text that must be read.

Responsive behaviour:

- From the `lg` breakpoint the sidebar is fixed at 216 px; below it, it becomes a top bar with a Menu toggle.
- The meeting grid drops to one column at phone width.
- The floor plan and elevation keep their scale and scroll horizontally inside their panel rather than shrinking, so measurements stay readable.
- The right column of the detail screen and the two persona columns stack under the plan below about 1,000 px.
- Wide tables scroll inside their frame with a 760 px minimum width.

## Next steps

1. Add the categorical palette and the Synthetics icon to the AeroAudio design system.
2. Add the new pipeline fields listed under Data contract.
3. Build the three screens against a real run, starting with the meeting detail.
4. Add table views for every chart, then test with keyboard and a screen reader.
