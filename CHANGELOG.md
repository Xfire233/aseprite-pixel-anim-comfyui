# Changelog

格式参照 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)。日期为本地开发日期。

## [0.3.0] - 2026-10-15

首个可分发版本：仓库本身就是一个 ComfyUI 自定义节点，克隆后运行 `install.py` 即可使用。

### 新增
- `scripts/install.py`：一键安装 Python 依赖、依赖节点和全部模型（断点续传 + sha256 校验），并打上 ComfyUI 核心补丁。RISA CUDA 编译失败时，工作流自动退回 `pytorch_attention`。
- `models.json`：所有模型的唯一清单（文件名、下载地址、目录、sha256、用途、许可）。
- `scripts/embed_model_urls.py`：把 `models.json` 写进工作流各加载节点的 `properties.models`。拖入工作流时，ComfyUI 前端会列出缺失模型并提供一键下载。
- `AsepritePoseRegionMask` 节点：把 DWPose 关键点转成骨盆/胸部羽化遮罩，支持多人和任意朝向。
- `scripts/paths.py`：统一管理路径，自动识别 ComfyUI 目录和 Aseprite 位置，也可用环境变量覆盖。
- README（使用说明 + 模型/技术清单）、`.env.example`、`scripts/charcard.example.json`、`pyproject.toml`、LICENSE。
- `validate_workflows.py` 新增检查：工作流里写死了 LLM `base_url` 或 `api_key` 时报错。

### 变更
- **Stage 1B 重写**，改为标准节点遮罩重绘：`DWPreprocessor` → `AsepritePoseRegionMask` → `SetLatentNoiseMask` + `KSampler`（Anima，denoise 0.70）→ `ImageCompositeMasked` → `RMBG`。遮罩外完全保留 Qwen 的输出，不再整图发糊。
- **骨盆遮罩改为沿"颈→髋"体轴放置**，不再参考膝盖：蹲姿时膝盖和髋差不多高，会把遮罩往上拉，盖不住短裤下半截。
- **Stage 2 动作轨迹改为按骨骼段刚体变形**：头和躯干保持刚性，四肢按关节旋转，修复了之前"整张图被压扁"的问题。
- **`solo_squat_cowgirl` / `solo_m_legs_sit` 骨骼的手臂改为外展**，同时改写提示词里"手放大腿"一类描述，减少手挡住胯部的情况。
- **默认不带任何 LLM 接口**：`base_url` 默认留空、模型列表为空，用户自填；不设时用本地模板编译。
- **默认角色卡改为 `none`**，从参考图自动识别外观；删除内置的个人角色卡预设。
- **目录调整**：工作流移到 `workflows/`；`.aseprite`/GIF/sheet 输出到 `ComfyUI/output/aseprite/`；骨骼库索引改为相对路径。
- **`gen.py`**：`--ref` 和 `--pose-file` 可以直接给任意路径（自动复制进 ComfyUI input）；`--card` 默认 `none`；Stage 1B 与工作流保持一致。

### 移除
- `AsepriteNSFWCharacterRefiner` 节点：它先把胯部平涂成肤色、再整图重采样，导致骨盆区细节只剩约 25%、全身线条变软。
- 写死的个人路径和私有 LLM 地址。

### 修复
- `AsepriteCharcardExtractor` 调用 `extract_charcard()` 时参数名对不上（此前点运行必定报错）。
- `gen.py` 在角色卡缺少 `comfy_ref` 字段时 KeyError。
- `extract_charcard.py` 没配置 LLM 时，会请求一个硬编码的地址。

## [0.2.0] - 2026-10-15

- Stage 1A 改用 Qwen-Image-Edit-2511 W4A8 ConvRot，配合 Lightning 4 步 LoRA（Bypass 挂载）和 RISA INT8 注意力，单帧约 3–13 秒。
- Stage 2 改用 Wan2.2 I2V 14B INT8 ConvRot 双专家，配合 LightX2V 4 步 LoRA，单段约 13 秒。
- 所有 VAE 解码改为分块，解决 16 GB 显卡的 WDDM 换页卡死。
- 新增 10 套 NSFW 骨骼姿势与动作轨迹库、角色卡提取器、LLM 提示词导演节点。
- ComfyUI 核心修复：多个 Bypass LoRA 叠加时互相覆盖。

## [0.1.0] - 2026-10-14

- 初版管线：Qwen-Image-Edit 逐帧换姿势 → 像素化（共享调色板、alpha 共识）→ Aseprite Lua 组装；新增 `validate_loop.py` 循环质量校验。
