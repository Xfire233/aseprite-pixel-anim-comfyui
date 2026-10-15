-- assemble.lua — 把一串帧 PNG 组装成 .aseprite（管线 S2/S4 批量组装，无 MCP 路线）
--
-- 用法（--script-param 必须在 --script 之前）:
--   Aseprite.exe -b ^
--     --script-param width=160 --script-param height=112 ^
--     --script-param frames=C:/path/frame1.png,C:/path/frame2.png ^
--     --script-param durations=100,120 ^
--     --script-param tag=loop ^
--     --script-param out=C:/path/char.aseprite ^
--     --script scripts/assemble.lua
--
-- 参数:
--   width/height  画布尺寸（默认 160x112）
--   frames        逗号分隔的帧 PNG 路径，按序对应帧 1..N（必填）
--   durations     逗号分隔的每帧毫秒数；缺省或不足补 100
--   tag           覆盖全部帧的 tag 名；缺省 loop
--   layer         图层名；缺省 frames
--   out           输出 .aseprite 路径（必填）

local p = app.params
local width = tonumber(p.width) or 160
local height = tonumber(p.height) or 112
local tagName = p.tag or "loop"
local layerName = p.layer or "frames"
local out = p.out
assert(out, "missing --script-param out=")
assert(p.frames, "missing --script-param frames=")

local function split(s)
  local t = {}
  for part in string.gmatch(s, "[^,]+") do t[#t + 1] = part end
  return t
end

local frames = split(p.frames)
local durations = p.durations and split(p.durations) or {}

local spr = Sprite(width, height)
local layer = spr.layers[1]
layer.name = layerName

for _ = 2, #frames do spr:newFrame() end

for i, path in ipairs(frames) do
  local img = Image{ fromFile = path }
  local cel = spr:newCel(layer, spr.frames[i])
  cel.image:drawImage(img, Point(0, 0))
  spr.frames[i].duration = (tonumber(durations[i]) or 100) / 1000.0
end

local tag = spr:newTag(1, #frames)
tag.name = tagName

spr:saveAs(out)
print("assemble.lua: saved " .. #frames .. " frames -> " .. out)
