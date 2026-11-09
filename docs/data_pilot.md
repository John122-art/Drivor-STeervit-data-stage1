# Steer-DrivoR data-generation pilot

Latest stage-2 update: the bulk launcher now uses `stage2/generate.py` with camera-local cleaning and one VQA call per image, without model re-description. Keep valid phrases and other cameras; only a genuinely unusable view becomes empty. See `stage2_data_dlc.md` and the updated `stage2_8gpu_dlc_command.txt`. Earlier pilot/repair sections below describe historical experiments; the stage-1 pipeline remains unchanged.

This repository contains an offline, small-batch data-generation pipeline. It prepares current-frame NAVSIM images and annotations, generates one English description per camera with Qwen-Drive-1.0, projects NAVSIM GT 3D boxes into each camera, and writes 14-pixel patch-grid soft targets. It does not modify DrivoR model or training code and does not start training.

## Environments and source code

The environments already installed for `DrivoR-main-copy` are reused; this pipeline does not install packages or alter those environments.

| Work | Environment | Source reused |
| --- | --- | --- |
| NAVSIM scene loading, calibration, projection, validation | `/mnt/zhaozc_workspace/envs/drivorpython` | NAVSIM and `DrivoR-main-copy/navsim/agents/drivoR/drivor_features.py` |
| Qwen image caption generation | `/mnt/zhaozc_workspace/envs/mot_vega` | Qwen-Drive-1.0 processor/VQA defaults and the VLM-only loading pattern in `DrivoR-main-copy/navsim/agents/qwen_text/backend.py` |

Qwen code and its local model checkpoint are read from `/mnt/zhaozc_workspace/project/LQH/Qwen-drive`. RoBERTa token counts use the local `/mnt/zhaozc_workspace/project/LQH/RoBERTa` tokenizer. NAVSIM data and maps are read from `DrivoR-main-copy/dataset`. The launcher sets offline Hugging Face mode and keeps any cache under this repository; it never downloads dependencies.

## Pilot outputs

The completed pilot lives in `outputs/data_pilot_20261107/`:

- `prepared/manifest.json`: 20 scenes from 20 logs, each with four current-frame camera images in `cam_f0`, `cam_b0`, `cam_l0`, `cam_r0` order. The VLM manifest contains image paths and hashes, not GT annotations.
- `prepared/ground_truth.json`: NAVSIM 3D boxes, names, scaled intrinsics and lidar-to-camera transforms. This file is consumed only by the projection stage.
- `captions_v3/scene_text.json`: final eight descriptions for the first two scenes (four views each), with categories aligned to the phrases.
- `captions_v3/audit.json`: raw model answers, prompts, filtering decisions, and token counts for review.
- `grounding_v3/masks/<scene_token>.npz`: for each captioned scene, `soft_mask` has shape `(4, 3936)`, `valid` has shape `(4,)`, and `camera_order` records the row order. Rows represent 48×82 patches at 14-pixel stride. Empty or unsupported categories have zero masks and `valid=false`.
- `grounding_v3/index.html`: image gallery with all 20 scenes' projected GT boxes, plus the eight captioned views and their patch-target overlays. Open this file in a browser to inspect the rendered images.
- `boxes_2d/index.html`: eight captioned views with only the selected 2D rectangles, displayed next to the corresponding patch targets. The exact `[xmin, ymin, xmax, ymax]` coordinates are exported to `boxes_2d/selected_boxes.json`; coordinates use the resized 1148×672 image. These rectangles are copied from the existing grounding projections without recomputing or changing the targets.
- `projection_audit/`: GT projection gallery created before captions were generated.
- `validation.json`: machine checks for original frame/image identity, calibration, checkpoint loading, text filtering and token length, masks, and gallery files.

The caption prompt describes visible appearance and image-relative position. It does not ask for fixed object counts. Future-motion terms and ambiguous ego-relative directions are filtered. Generation makes a single-image Qwen call per camera; no GT, history, navigation command or future frame enters the VLM. The GT projection clips 3D box edges at a 0.1 m camera near plane, skips boxes fully behind the camera, clips to image bounds, and normalizes the union coverage over the patch grid. Grounding selects every projected box within 50 m whose supported category appears in that view's caption.

## Review result and limits

All 20 four-camera projection sheets (80 views) were visually inspected. The camera order and projected geometry show no systematic shift in this dataset subset. The raw NAVSIM `lidar2ego` transform is identity for all 20 scenes, and the original image hashes and scaled calibration were rechecked. Three camera views required near-plane edge clipping. The two captioned scenes have eight successful captions and eight valid category masks. The longest final caption is 110 RoBERTa tokens, under the 128-token limit. Eleven geometry and text-processing unit checks passed.

These are category-level grounding targets: if a caption names a category, all eligible GT boxes of that category in the camera may be included. They can include occluded instances and do not identify the exact described instance. This pilot confirms the data path on a small sample; it does not establish caption quality across NAVSIM. The planned 50-sample single-image comparison against a general Qwen-VL model remains future work before bulk generation.

The final caption filter removed three motion-claim phrases from the rear view of the second scene and retained the remaining verified visual phrases. Raw outputs and removal reasons remain in the audit file.

## Reproduce the pilot stages

Each `--output` path must be new; scripts refuse to overwrite existing outputs. Use a fresh dated folder for a later pilot.

```bash
cd /mnt/zhaozc_workspace/project/LQH/Driv-calude_data/DrivoR-main

bash scripts/data_generation/run_data_pilot.sh prepare \
  --scenes 20 --output outputs/data_pilot_YYYYMMDD/prepared

bash scripts/data_generation/run_data_pilot.sh generate \
  --prepared outputs/data_pilot_YYYYMMDD/prepared \
  --scenes 2 --output outputs/data_pilot_YYYYMMDD/captions

bash scripts/data_generation/run_data_pilot.sh grounding \
  --prepared outputs/data_pilot_YYYYMMDD/prepared \
  --captions outputs/data_pilot_YYYYMMDD/captions \
  --output outputs/data_pilot_YYYYMMDD/grounding

bash scripts/data_generation/run_data_pilot.sh test

bash scripts/data_generation/run_data_pilot.sh validate \
  --prepared outputs/data_pilot_YYYYMMDD/prepared \
  --captions outputs/data_pilot_YYYYMMDD/captions \
  --grounding outputs/data_pilot_YYYYMMDD/grounding \
  --output outputs/data_pilot_YYYYMMDD/validation.json
```

The VLM stage needs the existing PPU SDK environment. It uses one image at a time and keeps the model in BF16. GridMask is not part of offline preparation, captioning, or projection. Training-time GridMask remains a separate future training decision.

## Driving-focused prompt and static phrase repair

The latest single-scene prompt trial uses `scripts/data_generation/prompts/scene_path_v3.txt`. It prioritizes vehicles and pedestrians with image-relative locations, visible lane/crosswalk relationships, road markings, traffic lights, and obstacles. Background buildings, advertising, and decorative structures are excluded unless they directly define or obstruct the road. The prompt aims for 60–80 English words for informative views; the existing 128-RoBERTa-token limit still applies.

Pass `--prompt-file scripts/data_generation/prompts/scene_path_v3.txt` to the generation command to select it. Existing default and prior prompt files are retained for reproducibility.

After schema validation, `caption_repair.py` identifies phrases containing unsupported motion/future/advice wording or ambiguous ego-relative positions. It reuses the same VLM and the same single current image to rewrite each affected object as a short static appearance/location description. Each rewrite must retain exactly one object and its category, pass the cleaner, and stay within 20 words. Other objects retain their order and wording. There are at most two rewrite attempts per affected phrase; unresolved phrases are recorded and remain subject to filtering. The final combined text is checked again against the 128-token limit.

The audit stores `initial_cleaning`, per-object `repairs`, raw rewrite responses, acceptance decisions, and whether a repaired object survived final token-budget filtering. `scene_text.json` contains only the final cleaned/repaired text and aligned categories. The preview page displays the final text and exposes repair history. Keyword checks do not prove factual accuracy; image review remains necessary.

## Navigation-conditioned descriptions

The latest navigation trial uses `scripts/data_generation/prompts/scene_navigation_v2.txt` and the generation flag `--with-navigation`. The first `v1` prompt is retained as a diagnostic because it produced excessively short noun lists. The latest prompt explicitly asks for descriptive phrases with image positions and visible road relationships. It retains one current image per camera and uses the current input navigation command to prioritize visible scene evidence relevant to the intended route. The output remains per-camera `text` and aligned `categories`; it describes current constraints for planning and does not output a future trajectory or an ego driving decision.

The new preparation manifest records `AgentInput.ego_statuses[-1].driving_command`, checks it against the same raw current-frame field, and records its decoded text. No future trajectory is used to calculate the command. Old manifests without this input are rejected in navigation mode rather than silently assigning a default. Unknown/all-zero commands remain unknown.

The command decoder is adapted from `/mnt/zhaozc_workspace/project/LQH/DrivoR-main-copy/navsim/agents/qwen_text/prompts.py`. Its first three NAVSIM one-hot classes are left, straight, right, also matching the [original RecogDrive input-command conversion](https://github.com/xiaomi-research/recogdrive/blob/main/navsim/planning/script/run_generate_dataset_pipeline.py#L884). The reference DrivoR implementation additionally handles the fourth/empty unknown case.

Qwen-Drive's local official `src/qwen_drive/scene.py` defines `NAV_COMMANDS` in a different order: `GO STRAIGHT`, `TURN LEFT`, `TURN RIGHT`. The generator imports that vocabulary directly and resolves commands by their words. It never treats the NAVSIM argmax as a Qwen command index. The prompt uses the official `Active navigation command: [...]` format. Single-view preprocessing continues to use official `QwenDriveProcessor.encode_vqa`; the full three-view/history planning interface is not invoked for this four-camera current-image description task.

Every phrase rewrite receives the same current command and current camera image as the initial caption. The prompt explicitly treats navigation as route intent, not evidence of unseen lane geometry, signs, right of way, or another actor's future actions. It retains conflicting visible obstacles or controls. Rear-view image coordinates remain image-relative even though navigation is ego-relative.

```bash
bash scripts/data_generation/run_data_pilot.sh prepare \
  --scenes 1 --output outputs/navigation_prompt_NEW/prepared

bash scripts/data_generation/run_data_pilot.sh generate \
  --prepared outputs/navigation_prompt_NEW/prepared \
  --output outputs/navigation_prompt_NEW/captions \
  --scenes 1 --with-navigation \
  --prompt-file scripts/data_generation/prompts/scene_navigation_v2.txt
```

The same scene with a different navigation input is a different caption-generation input. Cache reuse must check both the image identity and the recorded command, as well as prompt/model versions. Navigation-conditioned texts should be generated with the corresponding available current command for train/validation/test.

## Separate text generation for the two training stages

Stage 1 now has its own prompt, `scripts/data_generation/prompts/stage1_grounding_v1.txt`. Its purpose is simple visual grounding: short English phrases describing an object's appearance/type and image-relative position. It does not request navigation-conditioned scene analysis or planning relevance. This first grounding trial focuses on the five supported GT categories: vehicle, pedestrian, bicycle, traffic_cone, and barrier. Traffic controls and road-layout descriptions remain part of the richer second-stage description rather than receiving unsupported first-stage grounding targets.

Stage 1 is run **without** `--with-navigation`. A prepared manifest may contain a current input command for reuse by stage 2, but generation in stage 1 omits that command from both the initial question and every repair question. Metadata records `navigation_conditioned=false` and an empty `navigation_by_scene`; the audit records `navigation_command=null`.

```bash
bash scripts/data_generation/run_data_pilot.sh generate \
  --prepared outputs/stage1_grounding_pilot_20261107/prepared \
  --output outputs/stage1_grounding_NEW/captions \
  --scenes 1 \
  --prompt-file scripts/data_generation/prompts/stage1_grounding_v1.txt
```

The two stages can reuse original images, current GT boxes, and calibration, while keeping separate text caches. Generate masks from each cache's final cleaned categories. Shorter first-stage descriptions do not change the current class-union target rule into instance-level labels; describing a particular white car still does not uniquely associate it with that GT instance. This update changes offline data generation only; the two-stage training implementation remains future work.
