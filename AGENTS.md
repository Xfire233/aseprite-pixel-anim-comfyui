# Aseprite workflow — agent pointers

本地像素角色动画管线。先读 `SPEC.md` 顶部「现行定版 + 坑速查」，细节考古再往下翻。

## 启动与代码位置

- 本仓库**就是** ComfyUI 自定义节点（根目录 `__init__.py` + `web/`）。安装方式见 `README.md`：克隆进 `ComfyUI/custom_nodes/`，再用 ComfyUI 的 Python 跑 `scripts/install.py`。
- 本机开发：`ComfyUI/custom_nodes/comfyui_aseprite_pixelize` 是指向本仓库的目录联接（junction），只有一份代码，改完重启 ComfyUI 即生效。
- 本机启动 ComfyUI：在 ComfyUI 目录下用它自带的 Python 运行 `python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --enable-cors-header "*"`（ComfyUI 位置由 `scripts/paths.py` 自动识别，也可设 `ASEPRITE_WF_COMFY_ROOT`）。
- ComfyUI 核心补丁在 `patches/comfyui_core_bypass_lora.patch`（多个 `LoraLoaderBypassModelOnly` 叠加互相覆盖）；升级 ComfyUI 后重跑 `scripts/install.py` 会自动重新打上。
- 模型清单唯一来源是 `models.json`；改了工作流里的模型文件名后运行 `scripts/embed_model_urls.py`。
- 不进 git：`work/`、`output/`、`input/`、`scripts/charcard_*.json`、任何权重和 secret（见 `.gitignore`）。

## 当前进度与未决问题（2026-10-15）

- Stage 1B 已改为标准节点遮罩重绘（见下方入口）。6 个 seed 实测：遮罩外细节 0.999+；短裤残留 ≤20 像素 3/6；骨盆区清晰度达 0.67 仅 1/6（Anima 重画后细节偏少）。对照图：`work/diag_e2e_6seeds.png`、`work/diag_pose_mlegs.png`。
- 未验证：换参考图（磁盘上只有一个角色）、双人姿势（`duo_*`）的遮罩与重绘。
- 默认 `charcard=none`（不带个人角色卡）跑主工作流时，短裤残留明显（一次实测 5923 像素），未定位原因；疑为提示词缺具体服装描述。
- 角色卡提取节点已接入 3 个带导演的工作流（参考图 → 提取 → 导演 `charcard_file`，覆盖下拉框）；`char_name` 默认 `custom_char` 会被反复覆盖。仅用非人物图做过连线冒烟，真实角色图的提取质量未评估。
- GitHub 私有仓库：`Xfire233/aseprite-pixel-anim-comfyui`（`main`）。全新机器上完整跑 `install.py`（含 62 GB 模型）未做过。
- 骑乘/蹲起动作：Stage 2 由骨骼轨迹驱动（`_warp_image_by_pose_kps` 按骨骼段变形 + LowNoise Wan 精修），实测有上下起伏；手仍可能落在胯前（已把 `solo_squat_cowgirl` 手臂外展，未在多 seed 上复测）。
- 已知小问题：`scripts/validate_loop.py` 默认 `--adj-max` 曾从 9000 放宽到 10500，属放松检查而非修复。
- 生成图里的裸露部位质量，视觉模型拒绝评价，只能由用户目检。

## 入口

- 生成单帧/换姿精修：`python scripts/gen.py --ref <参考图路径> --pose-preset solo_m_legs_sit --phase "<姿势描述>" --anima-refine --prefix phase0/<名>`（`--card scripts/charcard_<名>.json` 可选）
- 提取新角色卡：`python scripts/extract_charcard.py --image <参考图路径> --name <角色名> --local-only`（落盘 `scripts/charcard_<名>.json`）
- 重建 NSFW 骨骼姿势库：`python scripts/build_nsfw_pose_library.py`（索引 `poses/nsfw_library/index.json`，总览图 `poses/nsfw_library/catalog.png`）
- 生成动作段（双入口）：
  1. **Web 前端**：将 `workflows/i2i_qwen_edit_wan_aseprite_web.json` 或 `workflows/anima_wan_aseprite_full_web.json` 拖入 `http://127.0.0.1:8188` 运行；产物在 `ComfyUI/output/aseprite/`
  2. **CLI 一键生成+像素化**：`python scripts/wan_gen.py --image <首帧> --prompt "<动作描述>" --prefix phase0/<名> --pixelize-out work/<名> --assemble output/<名>.aseprite --zoom work/<名>_zoom.png`
- 像素化+组装+校验：`python scripts/pixelize.py --frames ... --subsample 4 --pingpong --no-register --out work/<名> --assemble output/<名>.aseprite --zoom work/<名>_zoom.png` && `python scripts/validate_loop.py --frames work/<名>/frame*.png`
- 节点（根目录 `__init__.py`）：`AsepriteLLMPromptDirector`, `QwenEditFastConditioning`, `AsepriteNSFWPoseLibrary`, `AsepritePoseRegionMask`, `AsepriteCharcardExtractor`, `AsepriteStageBridge`, `AsepritePixelizeAssemble`；改完必须重启 ComfyUI。Stage 1B = `DWPreprocessor` → `AsepritePoseRegionMask` → `SetLatentNoiseMask` + `KSampler`(Anima, denoise 0.70) → `ImageCompositeMasked` → `RMBG`，衣物残留就重摇 Stage 1B 的 KSampler seed。每次改 `workflows/*_web.json` 后运行 `python scripts/validate_workflows.py`。
- 定稿锚帧（本机，未入库）：`work/canonical/`（正面 = `front_canon_c_fixed.png`）。
## 纪律（2026-10-14 复盘定版，违反即返工）

- **对照门**：每批生成的完成标准 = 三方对照图（参考/新图/旧图）+ 按当前使用的角色卡 `scripts/charcard_<名>.json` 逐项核对（服装状态、配饰位置、全身入框、手脚、逐帧清晰度）。未对照不报完成。服装词条逐字抄角色卡，禁止写入与角色卡状态冲突的描述。
- **尝试预算**：候选种子 ≤2、同一问题修复路径 ≤2 种；超了停下报告。动手前先报预计耗时。单帧生成 240s 硬超时（gen.py 已内置）。
- **生成红线**：只用双图（定稿裁剪+骨骼），禁止三图链式（本机间歇爆显存卡死 ≥15 分钟）；`cpu_offload=disable`；同一循环的帧必须同批同尺度生成（异尺度帧会被像素化共识碾碎）。
- **测速口径**：冷/热各一次且换输入（ComfyUI 节点缓存会假报快），报数注明口径。
