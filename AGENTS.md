# Aseprite workflow — agent pointers

本地像素角色动画管线。先读 `SPEC.md` 顶部「现行定版 + 坑速查」，细节考古再往下翻。

## 启动与代码位置

- 本仓库**就是** ComfyUI 自定义节点（根目录 `__init__.py` + `web/`）。安装方式见 `README.md`：克隆进 `ComfyUI/custom_nodes/`，再用 ComfyUI 的 Python 跑 `scripts/install.py`。
- 本机开发：`ComfyUI/custom_nodes/comfyui_aseprite_pixelize` 是指向本仓库的目录联接（junction），只有一份代码，改完重启 ComfyUI 即生效。
- 本机启动 ComfyUI：在 ComfyUI 目录下用它自带的 Python 运行 `python main.py --listen 127.0.0.1 --port 8188 --disable-auto-launch --enable-cors-header "*"`（ComfyUI 位置由 `scripts/paths.py` 自动识别，也可设 `ASEPRITE_WF_COMFY_ROOT`）。
- ComfyUI 核心补丁在 `patches/comfyui_core_bypass_lora.patch`（多个 `LoraLoaderBypassModelOnly` 叠加互相覆盖）；升级 ComfyUI 后重跑 `scripts/install.py` 会自动重新打上。
- 模型清单唯一来源是 `models.json`；改了工作流里的模型文件名后运行 `scripts/embed_model_urls.py`。
- 不进 git：`work/`、`output/`、`input/`、`scripts/charcard_*.json`、任何权重和 secret（见 `.gitignore`）。

## 当前进度与基线定版（2026-10-16 模块化与局部重跑定版）

- **Stage 1A 黄金脱衣 LoRA 配方**：`qwen_nsfw_adv` (0.80) + `Unchained-XXX` (0.72) + `Lightning-4steps` (1.0)。
- **Stage 1B 局部精修与 Regex 错位防御**：`AsepritePoseRegionMask` 定版为 `pelvis=True, chest=False, size=1.4, feather=8`，`denoise=0.70`。
- **Stage 2 真实 Diffusion 动力学恢复**：彻底删除底层的 `_warp_image_by_pose_kps` 仿射拉伸 hook 与 `_ACTIVE_TRAJ_LATENT` 采样覆盖，恢复 Wan2.2 真实的扩散物理动力学。
- **模块化 LLM 架构与动作生成**：`AsepriteLLMProvider` 统一输出 `LLM_CLIENT`；`AsepriteLLMPoseGenerator` 支持自然语言直接生成单帧及 9 帧连续 OpenPose 骨骼序列（解剖刚性校验防反折）；`AsepriteImageCritic` 负责多指与畸变打标及质量熔断门禁。
- **重绘前原图预览与 5 大阶段清晰布局**：工作流按 `【阶段 0】`、`【阶段 1A】`、`【阶段 1B】`、`【阶段 2】`、`【阶段 3】` 5 大阶段严格列式横向排布；新增 Node 62 `【阶段 1A 产物预览·重绘前原图】`；StageBridge 挂载 `🎲 仅重摇 1B`、`🎲 仅重摇 2` 与 `skip_stage2_video` 快速调试开关。
- **自动化回归套件**：`scripts/eval_pipeline_baseline.py` 与 `scripts/validate_workflows.py` 全部 100% PASS。
- **骑乘位大冲程与解剖一致性探索定论（2026-10-16 沉淀）**：
  1. **放弃单图 AI 脑补活塞大冲程**：Wan2.2 审查与平滑惩罚会导致运动被横向微晃吞没；4 步双锚点（First-Last-Frame）受 3D-VAE 时间压缩率 4 限制只会在首尾硬切跳变；
  2. **放弃同 Seed 独立出两帧做关键帧**：全局注意力发散导致静止背景（男伴/床单）变动率高达 60.81%，线条剧烈沸腾；
  3. **放弃 Qwen 自然语言全局下潜**：无 ControlNet 像素级约束，强行下潜 80~100px 会将下半身拉长融化为肉色泥潭；
  4. **局部 Inpainting 边界**：必须使用 DWPose 骨骼紧凑胶囊遮罩（~10% 画布），严禁手画粗糙大多边形（会吹成肉球气球）；适合做原地交合微调与脱衣精修，无法做全身大冲程；
  5. **动作节奏归宿**：采用 2D 动画解耦（面部绝对锁死 + 胯下局部微动），非线性冲击顿挫感（高位 200ms -> 急坠 70ms -> 坐底 220ms）通过 Aseprite 复制帧与帧时长（律表）在 30 秒内手动编排，不可本末倒置。
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
