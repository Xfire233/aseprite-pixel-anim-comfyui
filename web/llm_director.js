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
    const convertedSlots = new Set(["base_url", "api_key", "model", "charcard", "user_prompt", "pose_preset", "prompt_mode", "stage1_positive", "stage1_negative", "stage2_positive", "stage2_negative", "system_prompt"]);
    if (Array.isArray(node.inputs)) {
        for (let i = node.inputs.length - 1; i >= 0; i--) {
            if (convertedSlots.has(node.inputs[i]?.name)) {
                node.removeInput(i);
            }
        }
    }
    if (Array.isArray(node.widgets)) {
        node.widgets = node.widgets.filter((w) => w.name !== "system_prompt");
    }
    const poseW = node.widgets?.find((w) => w.name === "pose_preset");
    const modeW = node.widgets?.find((w) => w.name === "prompt_mode");
    const poseNode = app.graph?._nodes?.find((n) => n.type === "AsepriteNSFWPoseLibrary");
    const libPoseVal = poseNode?.widgets?.find((w) => w.name === "pose_preset")?.value;
    const libEnableVal = poseNode?.widgets?.find((w) => w.name === "enable_pose_control")?.value;
    if (poseW) {
        if (libEnableVal === false) {
            poseW.value = "none (不使用预设骨骼)";
        } else if (typeof poseW.value !== "string" || poseW.value.startsWith("compile_if_empty") || poseW.value.startsWith("auto_compile") || poseW.value.startsWith("manual_override")) {
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

function hookPromptDialog() {
    const CanvasClass = window.LGraphCanvas || (typeof app !== "undefined" && app.canvas?.constructor);
    if (CanvasClass && !CanvasClass._asepritePromptHooked) {
        CanvasClass._asepritePromptHooked = true;
        const origPrompt = CanvasClass.prototype.prompt;
        CanvasClass.prototype.prompt = function (title, value, callback, event) {
            const isApiKey = (title && typeof title === "string" && title.toLowerCase().includes("key")) ||
                             (event?.widget?.name && event.widget.name.toLowerCase().includes("key"));
            const res = origPrompt.apply(this, arguments);
            if (isApiKey) {
                setTimeout(() => {
                    const dialog = document.querySelector(".graphdialog");
                    const input = dialog?.querySelector("input.value");
                    if (input) {
                        input.type = "password";
                        input.setAttribute("autocomplete", "new-password");
                        if (input.value === "••••••••••••••••" || input.value === "__SAVED__") {
                            input.value = "";
                            input.placeholder = "(已保存API Key，直接输入可覆盖)";
                        }
                    }
                }, 10);
            }
            return res;
        };
    }
}

function maskApiKeyWidget(node) {
    const key = node.widgets?.find((w) => w.name === "api_key" || w.name === "key");
    if (!key || key._isMasked) return;
    key._isMasked = true;
    if (key.value && key.value !== "__SAVED__" && !/^[•*]+$/.test(key.value)) {
        key._realValue = key.value;
    } else {
        key._realValue = key._realValue || "";
    }

    const persistKey = async () => {
        const val = key._realValue;
        if (!val || val === "__SAVED__") return;
        if (key._savedValue !== val) {
            key._savedValue = val;
            key._savePromise = api.fetchApi("/aseprite_llm/save_key", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ api_key: val }),
            }).then((resp) => resp.json()).catch((err) => { key._savedValue = null; throw err; });
        }
        await key._savePromise;
    };

    if (key.inputEl) {
        key.inputEl.type = "password";
        key.inputEl.setAttribute("autocomplete", "new-password");
    }

    const origCallback = key.callback;
    key.callback = function (value) {
        if (typeof value === "string" && !/^[•*]+$/.test(value) && !value.startsWith("(空")) {
            key._realValue = value;
            persistKey().catch((err) => console.warn("[AsepriteLLM] save_key failed:", err));
        }
        return origCallback?.apply(this, arguments);
    };

    key.serializeValue = async () => {
        await persistKey();
        return key._realValue ? "__SAVED__" : "";
    };

    const origSerialize = node.onSerialize;
    node.onSerialize = function (data) {
        origSerialize?.apply(this, arguments);
        const index = this.widgets ? this.widgets.indexOf(key) : -1;
        if (Array.isArray(data?.widgets_values) && index >= 0) {
            data.widgets_values[index] = key._realValue ? "__SAVED__" : "";
        }
    };

    const origDraw = key.draw;
    key.draw = function (ctx, owner, width, y, height) {
        const val = key._realValue || key.value;
        const isSet = Boolean(val && val !== "(空 = 自动读已存Key)" && val !== "");
        const origVal = key.value;
        key.value = isSet ? "••••••••••••••••" : "(空 = 自动读已存Key)";
        try {
            return origDraw ? origDraw.call(this, ctx, owner, width, y, height) : undefined;
        } finally {
            key.value = origVal;
        }
    };
}

app.registerExtension({
    name: "aseprite.LLMPromptDirector",
    init() {
        hookPromptDialog();
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        hookPromptDialog();
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

        if (nodeData.name === "AsepriteLLMProvider") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;
                maskApiKeyWidget(this);
                const baseUrlW = this.widgets?.find((w) => w.name === "base_url");
                const apiKeyW = this.widgets?.find((w) => w.name === "api_key");
                const modelW = this.widgets?.find((w) => w.name === "model");

                const setKeyBtn = this.addWidget("button", "🔑 点击设置 / 修改 API Key", null, async () => {
                    const currentVal = apiKeyW?._realValue || "";
                    const userKey = window.prompt("请输入您的 API Key (OpenAI 兼容接口密钥，留空则清空使用本地配置)：", currentVal);
                    if (userKey !== null) {
                        const trimmed = userKey.trim();
                        if (apiKeyW) {
                            apiKeyW._realValue = trimmed;
                            apiKeyW.value = trimmed ? "••••••••••••••••" : "(空 = 自动读已存Key)";
                        }
                        try {
                            await api.fetchApi("/aseprite_llm/save_key", {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify({ api_key: trimmed }),
                            });
                            alert(trimmed ? "API Key 已成功更新并安全持久化保存！" : "API Key 已清空（将使用本地配置）。");
                        } catch (e) {
                            alert(`保存 API Key 异常: ${e}`);
                        }
                        app.graph.setDirtyCanvas(true, true);
                    }
                });
                setKeyBtn.serialize = false;

                const fetchBtn = this.addWidget("button", "🔄 从 Base URL 拉取模型列表", null, async () => {
                    const baseUrl = baseUrlW?.value?.trim() || "";
                    const apiKey = apiKeyW?._realValue || apiKeyW?.value || "";
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
                            if (modelW && modelW.options) {
                                modelW.options.values = data.models;
                                if (!data.models.includes(modelW.value)) modelW.value = data.models[0];
                                app.graph.setDirtyCanvas(true, true);
                            }
                            alert(`已成功拉取 ${data.models.length} 个模型！可在下方 model 下拉框直接选择。`);
                        } else {
                            alert(`拉取模型失败: ${data.error || "未知错误"}`);
                        }
                    } catch (err) {
                        alert(`请求异常: ${err}`);
                    }
                });
                fetchBtn.serialize = false;
                return r;
            };
            const origConf = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origConf ? origConf.apply(this, arguments) : undefined;
                maskApiKeyWidget(this);
                return r;
            };
            return;
        }

        if (nodeData.name === "AsepriteStageBridge") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;
                this.addWidget("button", "🎲 仅重摇 Stage 1B (Anima 局部重绘)", null, () => {
                    const node1B = app.graph?._nodes?.find((n) => n.id == 53 || (n.type === "KSampler" && n.title?.includes("1B")));
                    if (node1B && Array.isArray(node1B.widgets) && node1B.widgets.length > 0) {
                        node1B.widgets[0].value = Math.floor(Math.random() * 1e14);
                        app.graph.setDirtyCanvas(true, true);
                        app.queuePrompt(0);
                    } else {
                        alert("未找到 Stage 1B (Anima) KSampler 节点！");
                    }
                }).serialize = false;

                this.addWidget("button", "🎲 仅重摇 Stage 2 (Wan2.2 动作视频)", null, () => {
                    const node2 = app.graph?._nodes?.find((n) => n.id == 23 || n.type === "KSamplerAdvanced");
                    if (node2 && Array.isArray(node2.widgets) && node2.widgets.length > 1) {
                        node2.widgets[1].value = Math.floor(Math.random() * 1e14);
                        app.graph.setDirtyCanvas(true, true);
                        app.queuePrompt(0);
                    } else {
                        alert("未找到 Stage 2 (Wan2.2) KSampler 节点！");
                    }
                }).serialize = false;
                return r;
            };
            return;
        }

        if (nodeData.name === "RegexReplace") {
            const origConf = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origConf ? origConf.apply(this, arguments) : undefined;
                const pat = this.widgets?.find((w) => w.name === "regex_pattern");
                const rep = this.widgets?.find((w) => w.name === "replace");
                if (pat && (!pat.value || pat.value === "")) {
                    pat.value = "^[\\s\\S]*?(?=masterpiece)";
                }
                if (rep && rep.value === "true") {
                    rep.value = "";
                }
                return r;
            };
            return;
        }

        if (nodeData.name === "AsepriteLLMPromptDirector") {
            const origConf = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origConf ? origConf.apply(this, arguments) : undefined;
                maskApiKeyWidget(this);
                repairDirectorNode(this);
                return r;
            };
        }
        if (nodeData.name === "AsepriteNSFWPoseLibrary") {
            const origCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = origCreated ? origCreated.apply(this, arguments) : undefined;
                const poseW = this.widgets?.find((w) => w.name === "pose_preset");
                const enableW = this.widgets?.find((w) => w.name === "enable_pose_control");
                const syncToDirector = () => {
                    const dirNode = app.graph?._nodes?.find((n) => n.type === "AsepriteLLMPromptDirector");
                    if (dirNode) {
                        const dPose = dirNode.widgets?.find((w) => w.name === "pose_preset");
                        const dMode = dirNode.widgets?.find((w) => w.name === "prompt_mode");
                        if (dPose) {
                            dPose.value = (enableW?.value === false) ? "none (不使用预设骨骼)" : (poseW?.value || "none (不使用预设骨骼)");
                        }
                        if (dMode) dMode.value = "auto_compile (LLM编译并自动回填下方文本框)";
                        app.graph.setDirtyCanvas(true, true);
                    }
                };
                if (poseW) {
                    const origCb = poseW.callback;
                    poseW.callback = function (v) {
                        syncToDirector();
                        if (origCb) return origCb.apply(this, arguments);
                    };
                }
                if (enableW) {
                    const origEnableCb = enableW.callback;
                    enableW.callback = function (v) {
                        syncToDirector();
                        if (origEnableCb) return origEnableCb.apply(this, arguments);
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
            maskApiKeyWidget(this);

            const setKeyBtn = this.addWidget("button", "🔑 点击设置 / 修改 API Key", null, async () => {
                const currentVal = apiKeyWidget?._realValue || "";
                const userKey = window.prompt("请输入您的 API Key (OpenAI 兼容接口密钥，留空则清空使用本地配置)：", currentVal);
                if (userKey !== null) {
                    const trimmed = userKey.trim();
                    if (apiKeyWidget) {
                        apiKeyWidget._realValue = trimmed;
                        apiKeyWidget.value = trimmed ? "••••••••••••••••" : "(空 = 自动读已存Key)";
                    }
                    try {
                        await api.fetchApi("/aseprite_llm/save_key", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ api_key: trimmed }),
                        });
                        if (trimmed) {
                            alert("API Key 已成功设置并安全保存！");
                        } else {
                            alert("API Key 已清空（将使用本地环境变量或默认配置）。");
                        }
                    } catch (e) {
                        alert(`保存 API Key 异常: ${e}`);
                    }
                    app.graph.setDirtyCanvas(true, true);
                }
            });
            setKeyBtn.serialize = false;
            if (apiKeyWidget) {
                apiKeyWidget.mouse = function (event) {
                    if (event.type === "pointerdown" || event.type === "mousedown") {
                        setKeyBtn.callback();
                        return true;
                    }
                };
            }

            const fillPromptWidgets = (prompts) => {
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
