"""Unified LLM Provider module for ComfyUI Aseprite workflow.
Decouples API authentication, model dispatching, and multimodal requests from downstream nodes.
"""
from dataclasses import dataclass
import json
import os
import re
import urllib.request
import urllib.error
from typing import Optional, Dict, Any, Union

from scripts.paths import resolve_llm_key, LLM_BASE_URL, PROJECT_ROOT


@dataclass
class LLMClient:
    """Authenticated client handle passed between ComfyUI nodes via LLM_CLIENT socket."""
    base_url: str
    api_key: str
    default_model: str
    timeout: int = 45

    def get_effective_model(self, model_override: str = "") -> str:
        m = (model_override or "").strip()
        if m and m not in ("(使用 Provider 默认模型)", "none", "default"):
            return m
        return (self.default_model or "").strip()

    def request_text(
        self,
        sys_prompt: str,
        user_prompt: str,
        model_override: str = "",
        image_b64: str = "",
        temperature: float = 0.2,
    ) -> str:
        base = (self.base_url or "").strip().rstrip("/")
        if not base:
            raise RuntimeError("LLMProvider base_url is empty! Please configure an OpenAI-compatible base URL.")
        model = self.get_effective_model(model_override)
        if not model or model.startswith("(先填"):
            raise RuntimeError(f"No valid LLM model selected for request (current: '{model}')")

        resolved_key = resolve_llm_key(self.api_key)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "ComfyUI-AsepriteLLMProvider/1.0",
        }
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"

        prefer_responses = "argon" in model.lower() or model.lower().endswith("-high")
        endpoints = ["responses", "chat"] if prefer_responses else ["chat", "responses"]

        # Retry with multimodal image first (if present), then fallback to text-only if safety filter blocks
        img_candidates = [image_b64, ""] if image_b64 else [""]
        errors = []

        for img_url in img_candidates:
            for ep in endpoints:
                try:
                    if ep == "chat":
                        url = f"{base}/chat/completions"
                        payload = {
                            "model": model,
                            "messages": [
                                {"role": "system", "content": sys_prompt},
                                {
                                    "role": "user",
                                    "content": (
                                        [
                                            {"type": "image_url", "image_url": {"url": img_url}},
                                            {"type": "text", "text": user_prompt},
                                        ]
                                        if img_url
                                        else user_prompt
                                    ),
                                },
                            ],
                            "temperature": temperature,
                        }
                        if model.lower().endswith("-high") or "argon" in model.lower():
                            payload["reasoning_effort"] = "low"
                    else:
                        url = f"{base}/responses"
                        payload = {
                            "model": model,
                            "instructions": sys_prompt,
                            "input": (
                                [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "input_image", "image_url": img_url},
                                            {"type": "input_text", "text": user_prompt},
                                        ],
                                    }
                                ]
                                if img_url
                                else user_prompt
                            ),
                        }
                        if model.lower().endswith("-high") or "argon" in model.lower():
                            payload["reasoning"] = {"effort": "low"}

                    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
                    raw_resp = urllib.request.urlopen(req, timeout=self.timeout).read().decode("utf-8")
                    rj = json.loads(raw_resp)

                    # Extract text across OpenAI / Gemini / Zeabur schemas
                    text = ""
                    if isinstance(rj.get("choices"), list) and rj["choices"]:
                        msg = rj["choices"][0].get("message") or {}
                        c = msg.get("content")
                        if isinstance(c, str):
                            text = c.strip()
                        elif isinstance(c, list):
                            parts = [p.get("text", "") for p in c if isinstance(p, dict) and p.get("text")]
                            text = "\n".join(parts).strip()
                        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
                        if not text and isinstance(reasoning, str) and "{" in reasoning and "}" in reasoning:
                            text = reasoning.strip()
                    elif isinstance(rj.get("output_text"), str):
                        text = rj["output_text"].strip()
                    else:
                        for item in rj.get("output") or []:
                            if isinstance(item, dict):
                                for c in item.get("content") or []:
                                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                                        text = c["text"].strip()
                                        break

                    if text:
                        return text
                    errors.append(f"{ep}{'(img)' if img_url else ''} returned empty text")
                except Exception as e:
                    errors.append(f"{ep}{'(img)' if img_url else ''} error: {e}")

        raise RuntimeError(f"LLM request to {model} failed across all endpoints: {'; '.join(errors)}")

    def request_json(
        self,
        sys_prompt: str,
        user_prompt: str,
        model_override: str = "",
        image_b64: str = "",
        temperature: float = 0.2,
    ) -> dict:
        text = self.request_text(
            sys_prompt=sys_prompt,
            user_prompt=user_prompt,
            model_override=model_override,
            image_b64=image_b64,
            temperature=temperature,
        )
        # Strip markdown fences if present
        text_clean = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
        text_clean = re.sub(r"```\s*$", "", text_clean.strip(), flags=re.MULTILINE)
        m = re.search(r"\{[\s\S]*\}", text_clean)
        if not m:
            raise ValueError(f"Failed to find valid JSON object in LLM response:\n{text[:400]}")
        return json.loads(m.group(0))
