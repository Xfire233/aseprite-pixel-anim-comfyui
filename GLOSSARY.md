# Aseprite Workflow Domain Model

Core concepts and canonical terminology for the modular Aseprite pixel animation and image generation pipeline.

## Language

### LLM Subsystem

**LLM Provider**:
A centralized ComfyUI node emitting an authenticated `LLM_CLIENT` handle to distribute API access across downstream nodes without duplicating keys or endpoints.
_Avoid_: LLM Config, API Settings, LLM Source

**Prompt Director**:
A specialized compiler node that translates user actions, character cards, and pose states into stage-specific positive and negative prompts.
_Avoid_: Prompt Generator, Text Master, Director Node

**Image Critic**:
An automated perceptual QA node evaluating generated frames for anatomical defects (multi-finger, joint warping) and character identity fidelity against reference images.
_Avoid_: Verifier, Image Grader, Rater, Inspector

### Pose & Motion

**Pose Generator**:
A node that synthesizes 18-point OpenPose coordinates from natural language prompts via LLM with kinematic bone validation.
_Avoid_: Skeleton Drawer, Pose Prompt, Auto-Poser

**Pose Sequence**:
An ordered multi-frame batch of OpenPose keypoint arrays driving continuous motion diffusion in Stage 2 (I2V).
_Avoid_: Pose Video, Skeleton Animation, Motion Frames

**Anatomical Constraint**:
Kinematic and topological rules enforcing human limb length ratios and joint angle ranges on synthesized keypoints to eliminate physical deformities.
_Avoid_: Skeleton Fixer, Anti-distortion, Bone Clamp

### Pipeline Stages

**Stage 1A (Pose Morph)**:
The initial multi-modal I2I edit transforming character pose and costume while locking face and hair geometry.
_Avoid_: Qwen Pass, First Draft

**Stage 1B (Anatomical Refine)**:
Targeted masked inpainting on high-frequency anatomical regions (e.g. pelvis) to produce clean 2D linework.
_Avoid_: Anima Pass, Inpaint Fix

**Stage 2 (Motion Diffusion)**:
Diffusion-based video generation producing continuous frame sequences from the Stage 1 anchor frame.
_Avoid_: Wan Pass, Video Render

**Stage 3 (Pixelize & Assemble)**:
Quantization, color palette consensus, bounding box stabilization, and `.aseprite` sprite sheet assembly.
_Avoid_: Export Pass, Downscaler
