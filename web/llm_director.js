import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

function repairAssembleNode(node) {
    if (!node) return;
    const legacyInputs = new Set(["no_register", "band_cleanup", "stocking_tint", "no_sandal"]);
    if (Array.isArray(node.inputs)) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            if (legacyInputs.has(node.inputs[i]?.name)) {
                node.removeInput(i);
            }
        }
    }
    if (Array.isArray(node.widgets)) {
        node.widgets = node.widgets.filter((w) => !legacyInputs.has(w.name));
    }
    const setW = (name, validFn, defVal) => {
        const w = node.widgets?.find((item) => item.name === name);
        if (w && !validFn(w.value)) {
            w.value = defVal;
            if (w.inputEl) w.inputEl.value = String(defVal);
        }
    };
    setW("name", (v) => typeof v === "string" && v.trim().length > 0, "wan_i2i_pose");
    setW("canvas", (v) => typeof v === "string" && /^\d+x\d+$/.test(v), "128x224");
    setW("subsample", (v) => typeof v === "number" && v >= 0 && v <= 64, 4);
    setW("pingpong", (v) => typeof v === "boolean", true);
    setW("consensus", (v) => ["alpha", "none", "full"].includes(v), "alpha");
    setW("colors", (v) => typeof v === "number" && v >= 4 && v <= 256 && v !== 180, 32);
    setW("frame_duration_ms", (v) => typeof v === "number" && v >= 20 && v <= 2000, 180);
    setW("gif_scale", (v) => typeof v === "number" && v >= 1 && v <= 10, 4);
    setW("tag", (v) => typeof v === "string" && v.trim().length > 0, "idle_loop");
}

function repairDirectorNode(node) {
    if (!node) return;
    if (Array.isArray(node.inputs)) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            if (node.inputs[i]?.name === "system_prompt") {
                node.removeInput(i);
            }
        }
    }
    if (Array.isArray(node.widgets)) {
        node.widgets = node.widgets.filter((w) => w.name !== "system_prompt");
    }
    const poseW = node.widgets?.find((w) => w.name === "pose_preset");
    const modeW = node.widgets?.find((w) => w.name === "prompt_mode");
    // Sync with AsepriteNSFWPoseLibrary if present in graph
    const poseNode = app.graph?._nodes?.find((n) => n.type === "AsepriteNSFWPoseLibrary");
    const libPoseVal = poseNode?.widgets?.find((w) => w.name === "pose_preset")?.value;
    if (poseW) {
        if (typeof poseW.value !== "string" || poseW.value.startsWith("compile_if_empty") || poseW.value.startsWith("auto_compile") || poseW.value.startsWith("manual_override")) {
            poseW.value = libPoseVal || "solo_m_legs_sit (单人·M字开腿坐地抚腿娇喘)";
        } else if (libPoseVal && libPoseVal !== poseW.value) {
            poseW.value = libPoseVal;
        }
    }
    if (modeW) {
        if (typeof modeW.value !== "string" || !modeW.value.includes("compile") && !modeW.value.includes("manual")) {
            modeW.value = "auto_compile (LLM编译并自动回填下方文本框)";
        }
    }
}

app.registerExtension({
    name: "aseprite.LLMPromptDirector",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name === "AsepritePixelizeAssemble") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;
                setTimeout(() => repairAssembleNode(this), 20);
                return r;
            };
            const origConf = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origConf ? origConf.apply(this, arguments) : undefined;
                repairAssembleNode(this);
                return r;
            };
            return;
        }

        if (nodeData.name === "AsepriteNSFWPoseLibrary") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;
                const poseW = this.widgets?.find((w) => w.name === "pose_preset");
                if (poseW) {
                    const origCb = poseW.callback;
                    poseW.callback = function (v) {
                        const dirNode = app.graph?._nodes?.find((n) => n.type === "AsepriteLLMPromptDirector");
                        if (dirNode) {
                            const dPose = dirNode.widgets?.find((w) => w.name === "pose_preset");
                            const dMode = dirNode.widgets?.find((w) => w.name === "prompt_mode");
                            if (dPose) dPose.value = v;
                            if (dMode) dMode.value = "auto_compile (LLM编译并自动回填下方文本框)";
                            app.graph.setDirtyCanvas(true, true);
                        }
                        if (origCb) return origCb.apply(this, arguments);
                    };
                }
                return r;
            };
            return;
        }

        if (nodeData.name !== "AsepriteLLMPromptDirector") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
            setTimeout(() => repairDirectorNode(this), 20);

            const apiKeyWidget = this.widgets?.find((w) => w.name === "api_key");
            const baseUrlWidget = this.widgets?.find((w) => w.name === "base_url");
            const modelWidget = this.widgets?.find((w) => w.name === "model");

            // Auto-switch to auto_compile when user changes user_prompt, charcard, or pose_preset
            ["user_prompt", "charcard", "pose_preset"].forEach((wName) => {
                const w = this.widgets?.find((item) => item.name === wName);
                if (w) {
                    const origCb = w.callback;
                    w.callback = (v) => {
                        const modeW = this.widgets?.find((item) => item.name === "prompt_mode");
                        if (modeW) modeW.value = "auto_compile (LLM编译并自动回填下方文本框)";
                        if (wName === "pose_preset") {
                            const poseNode = app.graph?._nodes?.find((n) => n.type === "AsepriteNSFWPoseLibrary");
                            const pW = poseNode?.widgets?.find((item) => item.name === "pose_preset");
                            if (pW && pW.value !== v) pW.value = v;
                        }
                        if (origCb) return origCb.call(w, v);
                    };
                }
            });
            if (apiKeyWidget) {
                apiKeyWidget._realValue = apiKeyWidget.value || "";

                if (apiKeyWidget.inputEl) {
                    apiKeyWidget.inputEl.type = "password";
                }

                const origCallback = apiKeyWidget.callback;
                apiKeyWidget.callback = function (v) {
                    if (typeof v === "string" && !/^[•*]+$/.test(v) && !v.startsWith("(空")) {
                        apiKeyWidget._realValue = v;
                    } else if (v === "") {
                        apiKeyWidget._realValue = "";
                    }
                    if (origCallback) return origCallback.apply(this, arguments);
                };

                apiKeyWidget.serializeValue = () => {
                    const val = apiKeyWidget.value;
                    if (typeof val === "string" && val && !/^[•*]+$/.test(val) && !val.startsWith("(空")) {
                        apiKeyWidget._realValue = val;
                    }
                    return apiKeyWidget._realValue || "";
                };

                const origDraw = apiKeyWidget.draw;
                apiKeyWidget.draw = function (ctx, node, widgetWidth, y, H) {
                    const savedVal = apiKeyWidget.value;
                    if (typeof savedVal === "string" && savedVal && !/^[•*]+$/.test(savedVal) && !savedVal.startsWith("(空")) {
                        apiKeyWidget._realValue = savedVal;
                    }
                    apiKeyWidget.value = apiKeyWidget._realValue ? "••••••••••••••••" : "(空 = 自动读已存Key)";
                    try {
                        if (origDraw) {
                            return origDraw.call(this, ctx, node, widgetWidth, y, H);
                        }
                    } finally {
                        apiKeyWidget.value = apiKeyWidget._realValue || "";
                    }
                };
            }

            const fillPromptWidgets = (prompts) => {
                if (!Array.isArray(prompts) || prompts.length < 4) return;
                const names = ["stage1_positive", "stage1_negative", "stage2_positive", "stage2_negative"];
                names.forEach((n, idx) => {
                    const w = this.widgets?.find((item) => item.name === n);
                    if (w && typeof prompts[idx] === "string") {
                        w.value = prompts[idx];
                        if (w.inputEl) {
                            w.inputEl.value = prompts[idx];
                        }
                    }
                });
                app.graph.setDirtyCanvas(true, true);
            };

            const fetchBtn = this.addWidget("button", "🔄 从 Base URL 拉取模型列表", null, async () => {
                const baseUrl = baseUrlWidget?.value?.trim() || "";
                const apiKey = apiKeyWidget?._realValue || apiKeyWidget?.value || "";
                if (!baseUrl) {
                    alert("请先填写 base_url（例如 https://api.openai.com/v1）");
                    return;
                }
                try {
                    const resp = await api.fetchApi("/aseprite_llm/fetch_models", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
                    });
                    const data = await resp.json();
                    if (data.ok && Array.isArray(data.models) && data.models.length > 0) {
                        if (modelWidget && modelWidget.options) {
                            modelWidget.options.values = data.models;
                            if (!data.models.includes(modelWidget.value)) {
                                modelWidget.value = data.models[0];
                            }
                            app.graph.setDirtyCanvas(true, true);
                        }
                        alert(`已成功拉取 ${data.models.length} 个模型，可在 model 下拉框直接选择！`);
                    } else {
                        alert(`拉取模型失败: ${data.error || "未知错误"}`);
                    }
                } catch (err) {
                    alert(`请求异常: ${err}`);
                }
            });
            fetchBtn.serialize = false;

            const compileBtn = this.addWidget("button", "✨ 预编译提示词到下方4个文本框 (可查看/手改)", null, async () => {
                const getVal = (name) => this.widgets?.find((w) => w.name === name)?.value || "";
                const apiKey = apiKeyWidget?._realValue || apiKeyWidget?.value || "";
                try {
                    const resp = await api.fetchApi("/aseprite_llm/compile_preview", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            mode: "llm",
                            base_url: getVal("base_url"),
                            api_key: apiKey,
                            model: getVal("model"),
                            charcard: getVal("charcard"),
                            pose_preset: getVal("pose_preset"),
                            user_prompt: getVal("user_prompt"),
                            system_prompt: getVal("system_prompt"),
                        }),
                    });
                    const data = await resp.json();
                    if (Array.isArray(data.prompts)) {
                        fillPromptWidgets(data.prompts);
                    }
                    if (data.ok) {
                        alert("已编译并填入下方 4 个提示词框！您可以直接查看或手动修改文本框内容，然后点击运行。");
                    } else {
                        const modeW = this.widgets?.find((w) => w.name === "prompt_mode");
                        if (modeW) modeW.value = "manual_override (直接使用下方4个文本框，不调LLM)";
                        alert(`LLM 返回提示: ${data.error}\n\n已为您自动在下方 4 个文本框中展开角色卡与姿势预设英文模板，并切换为 manual_override 模式，您可直接手动修改下方文本框后运行！`);
                    }
                } catch (err) {
                    alert(`请求异常: ${err}`);
                }
            });
            compileBtn.serialize = false;

            const tplBtn = this.addWidget("button", "📋 展开角色卡模板到下方文本框 (不调LLM，直接手改)", null, async () => {
                const getVal = (name) => this.widgets?.find((w) => w.name === name)?.value || "";
                try {
                    const resp = await api.fetchApi("/aseprite_llm/compile_preview", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            mode: "template",
                            charcard: getVal("charcard"),
                            pose_preset: getVal("pose_preset"),
                            user_prompt: getVal("user_prompt"),
                        }),
                    });
                    const data = await resp.json();
                    if (Array.isArray(data.prompts)) {
                        fillPromptWidgets(data.prompts);
                        const modeW = this.widgets?.find((w) => w.name === "prompt_mode");
                        if (modeW) modeW.value = "manual_override (直接使用下方4个文本框，不调LLM)";
                    }
                } catch (err) {
                    alert(`展开模板异常: ${err}`);
                }
            });
            tplBtn.serialize = false;
            const origConf = this.onConfigure;
            this.onConfigure = function () {
                const rc = origConf ? origConf.apply(this, arguments) : undefined;
                repairDirectorNode(this);
                return rc;
            };

            const origOnExecuted = this.onExecuted;
            this.onExecuted = function (message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                if (message && Array.isArray(message.compiled_prompts)) {
                    fillPromptWidgets(message.compiled_prompts);
                }
            };

            return r;
        };
    },
});
