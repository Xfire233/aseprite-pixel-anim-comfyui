# Modular LLM Provider, Kinematic Pose Generation, and Perceptual Critic Architecture

## Context
Previously, LLM configurations (`base_url`, `api_key`, `model`) were tightly coupled inside the monolithic `AsepriteLLMPromptDirector` node. Adding LLM capabilities for OpenPose generation, role-card extraction, and visual quality evaluation would require duplicating credentials across multiple nodes, increasing maintenance overhead and leaking API keys in workflow files. Furthermore, prompt-only I2I editing without geometric anchors suffers from multi-finger and perspective distortion, and video diffusion (Wan2.2) lacks automated early-exit quality gates.

## Decision
1. **Decoupled LLM Provider (`AsepriteLLMProvider`)**: A single centralized provider node authenticates and manages endpoints/keys, emitting a structured `LLM_CLIENT` handle to all downstream consumer nodes (`PromptDirector`, `PoseGenerator`, `CharcardExtractor`, `ImageCritic`). Downstream nodes inherit default provider credentials while permitting local `model_override`.
2. **Kinematic OpenPose Synthesis (`AsepriteLLMPoseGenerator`)**: Synthesize OpenPose 18-point skeletons from natural language via LLM keyframe estimation (start, peak, end) coupled with local rigid kinematic interpolation (cosine/Bézier) and anatomical constraint clamping (limb length ratios, hinge angle limits) rather than frame-by-frame coordinate generation.
3. **Perceptual QA Gate (`AsepriteImageCritic`)**: Provide multimodal evaluation scoring anatomical completeness (hands/fingers, joint warping) and character identity fidelity, with an optional execution gate (`enable_gate`) to prevent running costly Stage 2 video diffusion on distorted Stage 1 frames.
