# Integrated hybrid pipeline: results

> **Update 2 (2026-11-03, see `COVERAGE.md`):** the garment length is now calibrated on 30 VITON-HD
> ground-truth pairs, and the studio render separates the garment's own coverage from the region that
> must change (`coverage="garment"`, now the default). Cropped garments over long tops are no longer
> extended to the old hem, and visible trousers are no longer repainted (38.5% → 0% outside the
> garment). Trade-off: hanging ties are often cut, and CatVTON sometimes adds a belt.
>
> **Update (2026-11-03, see `DEPTH.md`):** garment length now follows the product
> (proportion-aware mapping plus a garment-aware mask). On the 24 pairs this changed
> corr(length, product) from −0.17 to +0.68, colour match from 0.689 to 0.632, and old collar
> zone left from 1.7% to 2.4%. The tables below are the state *before* that change; the "before"
> images are kept in `results/studio_standard_hipmapped/`.

*Visionary · CMAI480 · 2026-11-03. This follows `AUDIT.md` (research and root-cause analysis). The
inputs are the same: 6 people × 4 garments, `eval/pairs.json`. All runs are local on an Apple M1 Pro
(16 GB), with no hosted service. The figures are in `results/figures/`; the per-pair numbers are in
each `results/<method>/metrics.csv`.*

## What changed

| Area | Change | Our own CV? |
|---|---|---|
| Instant render + live preview | Body parsing (SegFormer for photos, MediaPipe multiclass for live). **TPS edge fit** onto the person's clothing outline. Occlusion: face above the chin, hair and forearms stay in front. No garment on the background | TPS, bilinear remap, homography, inverse warp, blend: **ours**. Parsing: pretrained |
| Studio render | Local **hybrid**: our pose, homography/TPS footprint and parsing build the try-on mask. **CatVTON** (pretrained, Apple GPU) renders inside it. Lab colour correction toward the product photo. **Our feathered blend** pastes it back | Mask construction, colour correction, compositing: **ours**. Rendering: pretrained |
| Neckline leak | Root cause: SegFormer's "face" label includes the neck, so the old collar sat outside the mask. Fix: protect the face only above a chin line estimated from pose, grow the old-top region, and paste back with the soft edge *outside* the mask | ours |
| Long sleeves | The mask's arm corridors were as narrow as the bare arm, so the model painted short sleeves. Fix: corridors that may include background, down to the wrists when the garment is long-sleeved | ours |
| Garment anchors | Bug fix: a flared hem was counted as "sleeve", so short sleeves looked garment-length. Only sleeve pieces that reach armpit height now count | ours |
| Memory / speed | Sliced fp32 attention: **31.8 s / 5.5 GB** instead of 35.7 s / 9.9 GB (same output, mean pixel difference 0.28/255). Live parsing runs in a background thread. The TPS map is evaluated on an 8-px grid and upsampled with separable interpolation matrices | ours |
| MATLAB | `studioRender.m` plus a *Studio render* button in `TryOnApp` (calls the local server with `webwrite`) | — |

**Why MATLAB uses HTTP and not `py.*`** (verified with `file`): MATLAB R2023b on this Mac is `x86_64`
(Rosetta), and the venv's `libtorch` is `arm64`. The Apple-GPU back end needs a native arm64 process,
so the model cannot be loaded inside MATLAB. The server runs on the same machine.

## Results on all 24 pairs

| | spill on background ↓ | forearms covered ↓ | IoU with clothing region ↑ | face SSIM (above chin) ↑ | rest-of-image SSIM ↑ | garment colour match ↑ | old top unchanged ↓ | old **collar zone** unchanged ↓ (pairs > 10%) | time |
|---|---|---|---|---|---|---|---|---|---|
| classical (before) | 0.056 | 0.241 | 0.643 | 0.985 | 0.994 | 0.487 | 0.079 | 0.061 (6) | 0.20 s |
| **instant** (after) | **0.000** | **0.000** | **0.718** | **1.000** | 0.995 | 0.486 | 0.069 | 0.071 (7) | 0.21 s + parsing |
| instant without TPS | 0.000 | 0.000 | 0.707 | 1.000 | 0.995 | 0.477 | 0.083 | – | 0.17 s |
| hybrid PoC (before) | – | – | – | 1.000 | 0.997 | 0.510 | 0.035 | 0.096 (7) | ~30 s |
| **studio render** (after) | – | – | – | 0.999 | 0.996 | **0.689** | **0.020** | **0.017 (1)** | median **40 s** (32–74) |
| studio, no colour fix | – | – | – | 0.999 | 0.996 | 0.587 | 0.018 | – | – |

**Supported by these numbers:**
- **The neckline fix works.** The old collar zone left unchanged fell from 9.6% (7 pairs over 10%) to 1.7% (1 pair). The remaining pair (00035 + velvet) is a metric false positive: the velvet's dark V-lining has almost the same colour as her old navy top. It looks correct in `before_after_00035_00.jpg`.
- **Colour correction helps.** Garment colour match goes from 0.587 to 0.689 on the *same* generations, and improves 19 of 24 pairs.
- **The instant pipeline's occlusion rules work.** Spill onto background drops from 5.6% to 0, and forearms covered from 24% to 0. Face SSIM rises from 0.985 to 1.000 because collars no longer cover the chin.
- **TPS gives a small but consistent gain:** IoU from 0.707 to 0.718, and old top left uncovered by the garment from 8.8% to 7.1%, for ~40 ms.
- **The studio render keeps identity:** pixels outside the mask are the input photo (SSIM 0.996 for the rest of the image, 0.999 for the face).

**Caveats:**
- **The parser is also our evaluator.** The SegFormer parser that builds the masks also measures the results, so the occlusion metrics are partly self-consistent by construction. The visual sheets are the independent check.
- **Benchmark vs app attention mode.** The benchmark studio images were rendered with the "upcast" attention mode. The app now uses the "sliced" mode (same output within 0.28/255).
- **"Face" vs "face above chin".** The `ssim_face` column in the CSVs uses the parser's "face" label, which includes the neck. A new neckline legitimately changes the neck, so the table uses the face above the chin.

**Live preview** (640×480, idle M1 Pro, same frame streamed, server-side FPS): **10.8 FPS** with TPS and occlusion on, 14.1 FPS with both off. Before the background-parsing and TPS optimisations it was 3–7 FPS.

## Detail level (768×576, the 1024-trained "mix" checkpoint): rejected for the app

Measured on the 12 gingham and velvet pairs:

| | garment colour match | old top unchanged | collar zone | time |
|---|---|---|---|---|
| studio standard | **0.655** | 0.008 | 0.017 | ~40 s |
| studio detail | 0.469 | **0.001** | **0.004** | ~134 s |

Detail renders texture more sharply and clears necklines even better. But it **shifts garment colours**: the velvet comes out pale blue. Once (stock_man) it **changed the garment type**: the velvet wrap became a grey long-sleeved top. Native 1024×768 would need an 18 GiB attention buffer, more than this Mac has. The app therefore offers only Standard; Detail stays as an API option (`quality: "detail"`).

## Before / after (identical inputs)

`results/figures/before_after_<person>.jpg` has one sheet per person. Each row shows: person, garment, classical (before), instant (after), hybrid PoC (before), studio standard (after), studio detail (after, gingham and velvet only). The full grids are `grid_instant.jpg` and `grid_studio_standard.jpg`.

**Logos and prints** (`logo_test.jpg`): the benchmark garments have none, so two excluded VITON-HD tees were tested through the upload path:
- **Stüssy:** the large script print is reproduced and readable.
- **GUCCI:** the wordmark and green-red-green stripe are kept; the emblem is simplified; the small neck label is lost.
- **Scope:** this is 2 examples, not a measurement.

## Remaining failure cases

1. **01992 + colour-block polo:** a pale haze on the bare right shoulder next to the new sleeve. The model blends the light-blue yoke into the skin.
2. **stock_man (full-body photo, torso ≈ 200 px wide):** every render is soft, because the input itself is low-resolution where the garment goes. The model works at 512×384 on a crop, so it cannot add detail that is not there.
3. **Fine repeating patterns:** the gingham check becomes a fine texture, not a crisp check, at 512×384. Detail sharpens it but costs colour fidelity (see above).
4. **Small trims:** small details such as the red tee's tie are sometimes simplified or moved.
5. **Instant mode is still an overlay.** It cannot remove an old long sleeve that the new garment does not cover, or invent skin, folds or a new neckline. That is the classical ceiling found in the audit.
6. **Sleeveless garments in instant mode:** the anchor detector puts their "sleeve" points on the body sides, so the instant renderer's sleeve rotation is meaningless for them. The studio render is unaffected (it uses the catalogue's sleeve hint).
7. **Timing outliers:** under memory pressure (other apps open, or the Mac idle-sleeping), a render can take 2–5 minutes instead of ~40 s. Sliced attention lowered the peak from 9.9 to 5.5 GB, which removed the extreme case (an uploaded-garment render fell from 17 min to 32 s), but 16 GB is still the binding limit.

## Not verified here

- **The MATLAB side was not run.** MATLAB needs an interactive MathWorks sign-in, which this session cannot do. The server endpoint was tested with exactly the JSON `studioRender.m` builds (library garment, uploaded garment, and both error cases: HTTP 200/200/422/422). The MATLAB code itself was reviewed by hand only.
- **Live FPS is server-side**, measured by streaming one frame repeatedly. A real webcam adds capture and browser time.
- **Hosted models** (Leffa, FASHN, more IDM-VTON) were not re-run: the anonymous quota ran out, and the app no longer needs them.

## Reproduce

```bash
cd webapp
.venv/bin/python -m tests.test_core                     # 20 checks
.venv/bin/python -m eval.run_classical                  # before
.venv/bin/python -m eval.run_plus                       # instant (after); --no-tps for the ablation
caffeinate -i .venv/bin/python -m eval.run_studio standard
caffeinate -i .venv/bin/python -m eval.run_studio detail --only gingham_shirt,velvet_wrap_top
.venv/bin/python -m eval.score_outputs classical instant hybrid studio_standard_nofix studio_standard
.venv/bin/python -m eval.neckline classical instant hybrid studio_standard studio_detail
.venv/bin/python -m eval.make_figures
```
