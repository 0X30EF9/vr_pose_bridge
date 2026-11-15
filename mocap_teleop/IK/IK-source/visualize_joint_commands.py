#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import webbrowser
import xml.etree.ElementTree as ET
from typing import Any, Mapping, Sequence

from teleop_joint_schema import ARM_JOINT_NAMES, HAND_JOINT_ORDER, arm_joint_names, joint_names_from_motor_ids

DEFAULT_LOG = Path(__file__).resolve().parent / "log" / "mocap_ik_bridge.log"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "visualization" / "joint_commands.html"
ROBOT_URDF = (
    Path(__file__).resolve().parent
    / "models"
    / "walker_tienkung_ei_description"
    / "walker_tienkung_ei.urdf"
)
RELEVANT_SECTIONS = {"robot_command", "hand_ik_local_result"}


@dataclass(frozen=True)
class CommandSample:
    timestamp: float
    frame: int
    channel: str
    names: tuple[str, ...]
    positions: tuple[float, ...]
    published: bool
    topic: str
    diagnostics: Mapping[str, Any] | None = None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从遥操作日志生成零依赖的离线关节角 HTML 曲线",
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_LOG, help="mocap_ik_bridge.log 或纯 JSONL")
    parser.add_argument("--output", "-o", type=Path, default=DEFAULT_OUTPUT, help="生成的单文件 HTML")
    parser.add_argument("--session", choices=("latest", "all"), default="latest", help="默认只看日志中最近一次连续运行")
    parser.add_argument("--session-gap", type=float, default=10.0, help="相邻记录超过该秒数时视为新会话")
    parser.add_argument("--max-points", type=int, default=5000, help="每条轨迹最多嵌入的采样点")
    parser.add_argument("--open", action="store_true", help="生成后用默认浏览器打开")
    return parser.parse_args(argv)


def json_record_from_line(line: str) -> Mapping[str, Any] | None:
    start = line.find("{")
    if start < 0:
        return None
    try:
        record = json.loads(line[start:])
    except json.JSONDecodeError:
        return None
    return record if isinstance(record, Mapping) else None


def finite_positions(values: object) -> tuple[float, ...] | None:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return None
    try:
        positions = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not positions or not all(math.isfinite(value) for value in positions):
        return None
    return positions


def arm_names(data: Mapping[str, Any], count: int) -> tuple[str, ...] | None:
    explicit = data.get("names")
    if isinstance(explicit, Sequence) and not isinstance(explicit, (str, bytes)):
        names = tuple(str(name) for name in explicit)
        if len(names) == count:
            return names

    message = data.get("message")
    if isinstance(message, Mapping):
        commands = message.get("cmds")
        if isinstance(commands, Sequence) and not isinstance(commands, (str, bytes)):
            try:
                names = joint_names_from_motor_ids([int(command["name"]) for command in commands])
            except (KeyError, TypeError, ValueError):
                names = ()
            if len(names) == count:
                return names

    try:
        names = arm_joint_names(str(data.get("control_side", "both")))
    except ValueError:
        return None
    return names if len(names) == count else None


def command_sample(
    record: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None
) -> CommandSample | None:
    section = record.get("section")
    if section not in RELEVANT_SECTIONS:
        return None
    data = record.get("data")
    if not isinstance(data, Mapping):
        return None
    try:
        timestamp = float(record["time"])
        frame = int(record.get("frame", 0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(timestamp):
        return None

    if section == "robot_command":
        positions = finite_positions(data.get("positions"))
        if positions is None:
            return None
        names = arm_names(data, len(positions))
        if names is None:
            return None
        published = bool(data.get("enabled", False))
        return CommandSample(
            timestamp=timestamp,
            frame=frame,
            channel="arm_sent" if published else "arm_preview",
            names=names,
            positions=positions,
            published=published,
            topic=str(data.get("topic", "/arm/cmd_pos")),
            diagnostics=diagnostics,
        )

    positions = finite_positions(data.get("joint_positions"))
    order = data.get("joint_order", HAND_JOINT_ORDER)
    if positions is None or not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
        return None
    names = tuple(str(name) for name in order)
    if len(names) != len(positions):
        return None
    return CommandSample(
        timestamp=timestamp,
        frame=frame,
        channel="hand_local",
        names=names,
        positions=positions,
        published=bool(data.get("published", False)),
        topic="local_subprocess_pipe",
    )


def load_samples(path: Path) -> tuple[list[CommandSample], int]:
    samples: list[CommandSample] = []
    invalid_relevant = 0
    pending_direct_diagnostics: dict[int, Mapping[str, Any]] = {}
    with path.expanduser().resolve().open("r", encoding="utf-8", errors="replace") as source:
        for line in source:
            record = json_record_from_line(line)
            if record is None:
                continue
            section = record.get("section")
            if section == "direct_retarget_result":
                data = record.get("data")
                meta = data.get("meta") if isinstance(data, Mapping) else None
                if isinstance(meta, Mapping):
                    try:
                        pending_direct_diagnostics[int(record.get("frame", 0))] = meta
                    except (TypeError, ValueError):
                        pass
                continue
            if section not in RELEVANT_SECTIONS:
                continue
            diagnostics = None
            if section == "robot_command":
                try:
                    diagnostics = pending_direct_diagnostics.pop(int(record.get("frame", 0)), None)
                except (TypeError, ValueError):
                    diagnostics = None
            sample = command_sample(record, diagnostics)
            if sample is None:
                invalid_relevant += 1
            else:
                samples.append(sample)
    samples.sort(key=lambda sample: (sample.timestamp, sample.frame))
    return samples, invalid_relevant


def latest_session(samples: Sequence[CommandSample], gap_s: float) -> list[CommandSample]:
    if gap_s <= 0:
        raise ValueError("session-gap 必须大于 0")
    session_start = 0
    previous = samples[0]
    for index, sample in enumerate(samples[1:], 1):
        if sample.timestamp < previous.timestamp or sample.timestamp - previous.timestamp > gap_s:
            session_start = index
        previous = sample
    return list(samples[session_start:])


def sampled_indices(length: int, limit: int) -> list[int]:
    if limit <= 1:
        raise ValueError("max-points 必须大于 1")
    if length <= limit:
        return list(range(length))
    return sorted({round(index * (length - 1) / (limit - 1)) for index in range(limit)})


def _vector_attribute(element: ET.Element | None, name: str, default: str) -> list[float]:
    raw = default if element is None else element.attrib.get(name, default)
    values = [float(value) for value in raw.split()]
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError(f"URDF {name} 必须包含 3 个有限数值: {raw}")
    return values


def load_robot_model(path: Path = ROBOT_URDF) -> dict[str, Any]:
    """提取浏览器端正向运动学所需的最小 URDF 子集。"""
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"无法读取机器人 URDF: {path}") from exc

    requested = set(ARM_JOINT_NAMES)
    joints: dict[str, dict[str, Any]] = {}
    for element in root.findall("joint"):
        name = element.attrib.get("name", "")
        if name not in requested:
            continue
        origin = element.find("origin")
        parent = element.find("parent")
        child = element.find("child")
        axis = element.find("axis")
        limit = element.find("limit")
        if parent is None or child is None or limit is None:
            raise ValueError(f"URDF 关节定义不完整: {name}")
        joints[name] = {
            "name": name,
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "origin": _vector_attribute(origin, "xyz", "0 0 0"),
            "rpy": _vector_attribute(origin, "rpy", "0 0 0"),
            "axis": _vector_attribute(axis, "xyz", "1 0 0"),
            "lower": float(limit.attrib["lower"]),
            "upper": float(limit.attrib["upper"]),
        }

    missing = requested - joints.keys()
    if missing:
        raise ValueError(f"URDF 缺少双臂关节: {', '.join(sorted(missing))}")

    chains: dict[str, list[dict[str, Any]]] = {}
    for side, names in (("left", ARM_JOINT_NAMES[:7]), ("right", ARM_JOINT_NAMES[7:])):
        chain = [joints[name] for name in names]
        for parent_joint, child_joint in zip(chain, chain[1:]):
            if parent_joint["child"] != child_joint["parent"]:
                raise ValueError(
                    f"URDF {side} 臂运动链断开: {parent_joint['name']} -> {child_joint['name']}"
                )
        chains[side] = chain

    base_links = {chain[0]["parent"] for chain in chains.values()}
    if len(base_links) != 1:
        raise ValueError("URDF 左右臂不共享同一个躯干基座")
    return {
        "source": str(path.expanduser().resolve()),
        "base_link": base_links.pop(),
        "terminal_offset": [0.0, 0.0, -0.09],
        "chains": chains,
    }


def build_report(
    samples: Sequence[CommandSample],
    *,
    source: Path,
    invalid_relevant: int,
    max_points: int,
) -> dict[str, Any]:
    if not samples:
        raise ValueError("日志中没有可视化所需的 robot_command 或 hand_ik_local_result")
    start_time = min(sample.timestamp for sample in samples)
    grouped: dict[tuple[str, tuple[str, ...], bool, str], list[CommandSample]] = {}
    for sample in samples:
        key = (sample.channel, sample.names, sample.published, sample.topic)
        grouped.setdefault(key, []).append(sample)

    tracks = []
    for (channel, names, published, topic), track_samples in grouped.items():
        indices = sampled_indices(len(track_samples), max_points)
        tracks.append(
            {
                "channel": channel,
                "names": list(names),
                "published": published,
                "topic": topic,
                "original_count": len(track_samples),
                "samples": [
                    [
                        round(track_samples[index].timestamp - start_time, 6),
                        track_samples[index].frame,
                        list(track_samples[index].positions),
                        track_samples[index].diagnostics,
                    ]
                    for index in indices
                ],
            }
        )

    sent_count = sum(1 for sample in samples if sample.channel == "arm_sent")
    preview_count = sum(1 for sample in samples if sample.channel == "arm_preview")
    hand_count = sum(1 for sample in samples if sample.channel == "hand_local")
    return {
        "meta": {
            "source": str(source.expanduser().resolve()),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "start_time": datetime.fromtimestamp(start_time).astimezone().isoformat(timespec="milliseconds"),
            "duration_s": round(max(sample.timestamp for sample in samples) - start_time, 6),
            "sample_count": len(samples),
            "arm_sent_count": sent_count,
            "arm_preview_count": preview_count,
            "hand_local_count": hand_count,
            "invalid_relevant_lines": invalid_relevant,
            "joint_unit": "rad",
        },
        "robot_model": load_robot_model(),
        "tracks": tracks,
    }


def render_html(report: Mapping[str, Any]) -> str:
    payload = json.dumps(report, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__REPORT_DATA__", payload)


def write_report(report: Mapping[str, Any], output: Path) -> Path:
    target = output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_html(report), encoding="utf-8")
    return target


def summarize(report: Mapping[str, Any], output: Path) -> str:
    meta = report["meta"]
    return (
        f"VISUALIZATION_WRITTEN {output}\n"
        f"samples={meta['sample_count']} duration={meta['duration_s']:.3f}s "
        f"arm_sent={meta['arm_sent_count']} arm_preview={meta['arm_preview_count']} "
        f"hand_local={meta['hand_local_count']} invalid={meta['invalid_relevant_lines']}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.input.expanduser().resolve()
    if not source.is_file():
        print(f"输入日志不存在: {source}", file=sys.stderr)
        return 2
    try:
        samples, invalid_relevant = load_samples(source)
        if args.session == "latest" and samples:
            samples = latest_session(samples, args.session_gap)
        report = build_report(
            samples,
            source=source,
            invalid_relevant=invalid_relevant,
            max_points=args.max_points,
        )
        output = write_report(report, args.output)
    except (OSError, ValueError) as exc:
        print(f"生成可视化失败: {exc}", file=sys.stderr)
        return 1
    print(summarize(report, output))
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>遥操作关节角离线报告</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2d;--line:#29344d;--text:#e8edf7;--muted:#96a3ba;--accent:#5eead4;--sent:#34d399;--preview:#fbbf24;--hand:#60a5fa;--left:#5eead4;--right:#fb7185}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#17203a 0,var(--bg) 36%);color:var(--text);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{width:min(1500px,96vw);margin:0 auto;padding:28px 0 50px}h1{font-size:clamp(24px,3vw,38px);margin:0 0 6px}.sub{color:var(--muted);word-break:break-all}
.summary,.controls{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}.metric,.controls label{background:rgba(20,27,45,.88);border:1px solid var(--line);border-radius:12px;padding:10px 14px}.metric b{display:block;font-size:20px}.metric span{color:var(--muted)}
.controls{position:sticky;top:0;z-index:3;padding:12px;background:rgba(11,16,32,.9);backdrop-filter:blur(10px);border:1px solid var(--line);border-radius:14px}.controls label{padding:7px 10px;margin:0}.controls select,.pose-controls select{margin-left:7px;background:#0f172a;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:4px 7px}
.notice{padding:12px 14px;border-left:3px solid var(--accent);background:#111a2c;border-radius:7px;color:#cbd5e1;margin:14px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px}.card,.pose-panel{position:relative;background:rgba(20,27,45,.9);border:1px solid var(--line);border-radius:14px;padding:14px;min-width:0}.card h2{font:600 14px ui-monospace,SFMono-Regular,monospace;margin:0;overflow:hidden;text-overflow:ellipsis}.tag{float:right;border-radius:999px;padding:2px 8px;font-size:11px}.arm_sent .tag{background:#123b31;color:#6ee7b7}.arm_preview .tag{background:#493513;color:#fcd34d}.hand_local .tag{background:#172f53;color:#93c5fd}.stats{color:var(--muted);font-size:12px;margin:4px 0 8px}canvas{display:block;width:100%;height:190px;touch-action:none}.tip{display:none;position:absolute;pointer-events:none;z-index:2;background:#020617e8;border:1px solid #475569;border-radius:7px;padding:5px 8px;font:12px ui-monospace,monospace;white-space:nowrap}.empty{text-align:center;color:var(--muted);padding:50px}.hidden{display:none}
.pose-panel{margin:20px 0;padding:18px}.pose-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px}.pose-head h2{margin:0;font-size:20px}.pose-head span{color:var(--muted);font-size:12px}.pose-layout{display:grid;grid-template-columns:minmax(420px,1.5fr) minmax(260px,.75fr);gap:16px;margin-top:12px}.robot-stage{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:12px;background:radial-gradient(circle at 50% 42%,#1c2942 0,#0a1120 70%)}#robotCanvas{height:510px;cursor:grab}#robotCanvas.dragging{cursor:grabbing}.view-buttons{position:absolute;left:10px;top:10px;z-index:2;display:flex;gap:5px}.view-buttons button{border:1px solid #3a4967;background:#0b1323d9;color:#cbd5e1;border-radius:7px;padding:5px 9px;cursor:pointer}.view-buttons button.active{border-color:var(--accent);color:var(--accent);background:#12312f}.stage-help{position:absolute;left:12px;bottom:9px;color:#718096;font-size:11px;pointer-events:none}.pose-side{display:flex;flex-direction:column;min-width:0}.pose-status{padding:10px 12px;border-radius:9px;background:#0d1526;border:1px solid var(--line);margin-bottom:10px}.pose-status b{display:block;font-size:18px}.pose-status span{display:block;color:var(--muted)}.pose-status .shape{color:var(--accent);font-weight:700;margin-top:5px}.pose-status .bad-shape{color:#fbbf24}.joint-readout{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:1px 10px;font:12px ui-monospace,SFMono-Regular,monospace;overflow:auto;max-height:405px}.joint-readout div{padding:5px 7px;background:#10192a;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.joint-readout .side-left{border-left:2px solid var(--left)}.joint-readout .side-right{border-left:2px solid var(--right)}.joint-readout .limit{color:#fbbf24;font-weight:700}.pose-controls{display:flex;align-items:center;flex-wrap:wrap;gap:9px;margin-top:14px}.pose-controls button{border:1px solid #3a4967;background:#17233a;color:var(--text);border-radius:8px;padding:7px 11px;cursor:pointer}.pose-controls button:hover{border-color:var(--accent)}.pose-controls label{color:var(--muted)}#poseTrack{max-width:320px}#poseTime{flex:1;min-width:220px;accent-color:var(--accent)}#poseClock{font:12px ui-monospace,monospace;min-width:145px;text-align:right}.legend{display:flex;gap:14px;color:var(--muted);font-size:12px;margin-top:8px}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.legend .left{background:var(--left)}.legend .right{background:var(--right)}
@media(max-width:800px){.pose-layout{grid-template-columns:1fr}#robotCanvas{height:430px}.joint-readout{max-height:260px}.controls{position:static}}@media(max-width:500px){.grid{grid-template-columns:1fr}main{width:94vw}.pose-layout{display:block}#robotCanvas{height:370px}.pose-side{margin-top:12px}.pose-controls>*{max-width:100%}}
</style>
</head>
<body><main>
<h1>遥操作关节角离线报告</h1><div class="sub" id="source"></div>
<div class="summary" id="summary"></div>
<div class="notice" id="notice"></div>
<div class="controls">
<label>单位<select id="unit"><option value="deg">度 (°)</option><option value="rad">弧度 (rad)</option></select></label>
<label><input type="checkbox" data-channel="arm_sent" checked> 已发送双臂</label>
<label><input type="checkbox" data-channel="arm_preview" checked> 未发送预览</label>
<label><input type="checkbox" data-channel="hand_local" checked> 本地手部 IK</label>
</div>
<section class="pose-panel" id="posePanel">
<div class="pose-head"><h2>机器人 3D 姿态</h2><span>URDF 运动链 · 腰部以上固定 · 非网格渲染</span></div>
<div class="pose-layout">
<div class="robot-stage"><div class="view-buttons" id="viewButtons"><button type="button" data-view="front">正视</button><button type="button" data-view="side">侧视</button><button type="button" data-view="top">俯视</button><button type="button" data-view="3d" class="active">3D</button></div><canvas id="robotCanvas"></canvas><div class="stage-help">世界坐标：X 前 / Y 左 / Z 上 · 拖动旋转 · 滚轮缩放 · 双击恢复 3D</div></div>
<div class="pose-side"><div class="pose-status" id="poseStatus"></div><div class="joint-readout" id="jointReadout"></div><div class="legend"><span><i class="left"></i>左臂</span><span><i class="right"></i>右臂</span><span>黄色关节：接近 URDF 边界</span></div></div>
</div>
<div class="pose-controls">
<button id="posePlay" type="button">播放</button><button id="poseStepBack" type="button" title="上一帧">−1 帧</button><button id="poseStepForward" type="button" title="下一帧">+1 帧</button>
<label>命令<select id="poseTrack"></select></label><label>速度<select id="poseSpeed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
<input id="poseTime" type="range" min="0" max="0" step="1" value="0"><output id="poseClock"></output>
</div>
</section>
<section class="grid" id="charts"></section>
</main>
<script>
const REPORT=__REPORT_DATA__;
const COLORS={arm_sent:'#34d399',arm_preview:'#fbbf24',hand_local:'#60a5fa'};
const LABELS={arm_sent:'已发送',arm_preview:'未发送',hand_local:'本地未发送'};
const SIDE_COLORS={left:'#5eead4',right:'#fb7185'};
const VIEW_PRESETS={front:{yaw:0,pitch:0,zoom:650},side:{yaw:Math.PI/2,pitch:0,zoom:650},top:{yaw:0,pitch:-Math.PI/2,zoom:650},'3d':{yaw:-.62,pitch:.18,zoom:650}};
const state={unit:'deg',channels:new Set(Object.keys(COLORS)),charts:[],poseTracks:[],poseTrack:null,poseIndex:0,playing:false,playWall:0,playTime:0,view:{...VIEW_PRESETS['3d']},viewName:'3d',drag:null};
const fmt=n=>Number(n).toLocaleString('zh-CN',{maximumFractionDigits:3});
const el=id=>document.getElementById(id);
function metric(value,label){return `<div class="metric"><b>${value}</b><span>${label}</span></div>`}
function setup(){
  const m=REPORT.meta;
  el('source').textContent=`数据源：${m.source} · 起始：${m.start_time} · 生成：${m.generated_at}`;
  el('summary').innerHTML=metric(m.duration_s.toFixed(2)+' s','时长')+metric(m.arm_sent_count,'实际发送帧')+metric(m.arm_preview_count,'未发送预览帧')+metric(m.hand_local_count,'手部本地帧');
  const warnings=[];
  if(!m.arm_sent_count)warnings.push('本会话没有 enabled=true 的双臂命令；黄色轨迹只是计算预览，未发送给机器人。');
  if(m.hand_local_count)warnings.push('蓝色手部轨迹来自 Revo2 本地 IK；当前工程没有手部发布接口，不能视为机器人已接收。');
  if(m.invalid_relevant_lines)warnings.push(`${m.invalid_relevant_lines} 条相关日志格式无效，已跳过。`);
  el('notice').textContent=warnings.join(' ')||'绿色轨迹来自发布边界且 enabled=true，表示记录到的实际双臂下发命令。';
  REPORT.tracks.forEach((track,ti)=>track.names.forEach((name,ji)=>addChart(track,ti,ji,name)));
  el('unit').addEventListener('change',event=>{state.unit=event.target.value;redrawAll();renderPose()});
  document.querySelectorAll('[data-channel]').forEach(box=>box.addEventListener('change',()=>{box.checked?state.channels.add(box.dataset.channel):state.channels.delete(box.dataset.channel);filterCharts()}));
  setupPose();
  window.addEventListener('resize',redrawAll);
}
function addChart(track,ti,ji,name){
  const card=document.createElement('article');
  card.className=`card ${track.channel}`;card.dataset.channel=track.channel;
  card.innerHTML=`<span class="tag">${LABELS[track.channel]}</span><h2 title="${name}">${name}</h2><div class="stats"></div><canvas></canvas><div class="tip"></div>`;
  el('charts').appendChild(card);
  const chart={card,canvas:card.querySelector('canvas'),tip:card.querySelector('.tip'),track,ti,ji,name};
  state.charts.push(chart);
  chart.canvas.addEventListener('pointermove',event=>hover(chart,event));
  chart.canvas.addEventListener('pointerleave',()=>chart.tip.style.display='none');
  draw(chart);
}
function values(chart){const scale=state.unit==='deg'?180/Math.PI:1;return chart.track.samples.map(sample=>[sample[0],sample[2][chart.ji]*scale])}
function bounds(points){let lo=Infinity,hi=-Infinity;for(const point of points){lo=Math.min(lo,point[1]);hi=Math.max(hi,point[1])}const pad=Math.max((hi-lo)*.1,state.unit==='deg'?.25:.005);return[lo-pad,hi+pad]}
function draw(chart){
  if(chart.card.classList.contains('hidden'))return;
  const c=chart.canvas,rect=c.getBoundingClientRect(),dpr=devicePixelRatio||1;
  c.width=Math.max(1,Math.round(rect.width*dpr));c.height=Math.max(1,Math.round(rect.height*dpr));
  const x=c.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);
  const W=rect.width,H=rect.height,L=48,R=10,T=10,B=25,p=values(chart),range=bounds(p),lo=range[0],hi=range[1],t0=p[0][0],t1=p[p.length-1][0],dx=Math.max(t1-t0,.001);
  x.clearRect(0,0,W,H);x.strokeStyle='#29344d';x.fillStyle='#96a3ba';x.font='11px system-ui';x.lineWidth=1;
  for(let i=0;i<4;i++){const y=T+(H-T-B)*i/3,val=hi-(hi-lo)*i/3;x.beginPath();x.moveTo(L,y);x.lineTo(W-R,y);x.stroke();x.fillText(fmt(val),3,y+4)}
  x.fillText(fmt(t0)+'s',L,H-6);x.fillText(fmt(t1)+'s',W-R-42,H-6);x.strokeStyle=COLORS[chart.track.channel];x.lineWidth=1.7;x.beginPath();
  p.forEach((point,i)=>{const px=L+(point[0]-t0)/dx*(W-L-R),py=T+(hi-point[1])/(hi-lo)*(H-T-B);i?x.lineTo(px,py):x.moveTo(px,py)});x.stroke();
  const raw=p.map(point=>point[1]);chart.card.querySelector('.stats').textContent=`min ${fmt(Math.min(...raw))} · max ${fmt(Math.max(...raw))} · Δ ${fmt(Math.max(...raw)-Math.min(...raw))} ${state.unit==='deg'?'°':'rad'} · ${chart.track.original_count} 帧`;
  chart._plot={L,R,T,B,W,H,p,lo,hi,t0,t1,dx};
}
function hover(chart,event){
  const q=chart._plot;if(!q)return;
  const rect=chart.canvas.getBoundingClientRect(),mx=event.clientX-rect.left,target=q.t0+Math.max(0,Math.min(1,(mx-q.L)/(q.W-q.L-q.R)))*q.dx;
  let best=0;for(let i=1;i<q.p.length;i++)if(Math.abs(q.p[i][0]-target)<Math.abs(q.p[best][0]-target))best=i;
  const sample=chart.track.samples[best],point=q.p[best];chart.tip.textContent=`t=${fmt(point[0])}s · frame=${sample[1]} · ${fmt(point[1])}${state.unit==='deg'?'°':' rad'}`;
  chart.tip.style.display='block';chart.tip.style.left=Math.min(q.W-210,Math.max(5,mx+10))+'px';chart.tip.style.top='45px';
  if(chart.track.channel.startsWith('arm_'))selectPoseTrack(chart.ti,best);
}
function setupPose(){
  state.poseTracks=REPORT.tracks.map((track,index)=>({track,index})).filter(item=>item.track.channel.startsWith('arm_')&&item.track.samples.length);
  if(!state.poseTracks.length){el('posePanel').classList.add('hidden');return}
  const trackSelect=el('poseTrack');
  state.poseTracks.forEach(item=>{const option=document.createElement('option');option.value=item.index;const sides=armSides(item.track.names);option.textContent=`${LABELS[item.track.channel]} · ${sides} · ${item.track.original_count} 帧`;trackSelect.appendChild(option)});
  const initial=state.poseTracks.find(item=>item.track.channel==='arm_sent')||state.poseTracks[0];
  trackSelect.value=String(initial.index);state.poseTrack=initial;
  trackSelect.addEventListener('change',()=>selectPoseTrack(Number(trackSelect.value),0));
  el('poseTime').addEventListener('input',()=>{stopPose();state.poseIndex=Number(el('poseTime').value);renderPose()});
  el('posePlay').addEventListener('click',togglePose);
  el('poseStepBack').addEventListener('click',()=>stepPose(-1));el('poseStepForward').addEventListener('click',()=>stepPose(1));
  document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>setView(button.dataset.view)));
  const canvas=el('robotCanvas');
  canvas.addEventListener('pointerdown',event=>{state.viewName='';updateViewButtons();state.drag={x:event.clientX,y:event.clientY,yaw:state.view.yaw,pitch:state.view.pitch};canvas.setPointerCapture(event.pointerId);canvas.classList.add('dragging')});
  canvas.addEventListener('pointermove',event=>{if(!state.drag)return;state.view.yaw=state.drag.yaw+(event.clientX-state.drag.x)*.008;state.view.pitch=Math.max(-1.1,Math.min(1.1,state.drag.pitch+(event.clientY-state.drag.y)*.006));renderPose()});
  canvas.addEventListener('pointerup',()=>{state.drag=null;canvas.classList.remove('dragging')});canvas.addEventListener('pointercancel',()=>{state.drag=null;canvas.classList.remove('dragging')});
  canvas.addEventListener('wheel',event=>{event.preventDefault();state.view.zoom=Math.max(300,Math.min(1200,state.view.zoom*Math.exp(-event.deltaY*.001)));renderPose()},{passive:false});
  canvas.addEventListener('dblclick',()=>setView('3d'));
  configurePoseRange();renderPose();
}
function updateViewButtons(){document.querySelectorAll('[data-view]').forEach(button=>button.classList.toggle('active',button.dataset.view===state.viewName))}
function setView(name){if(!VIEW_PRESETS[name])return;state.view={...VIEW_PRESETS[name]};state.viewName=name;updateViewButtons();renderPose()}
function armSides(names){const left=names.some(name=>name.includes('_l_joint')),right=names.some(name=>name.includes('_r_joint'));return left&&right?'双臂':left?'左臂':'右臂'}
function selectPoseTrack(reportIndex,index){
  const selected=state.poseTracks.find(item=>item.index===Number(reportIndex));if(!selected)return;
  stopPose();state.poseTrack=selected;el('poseTrack').value=String(selected.index);configurePoseRange();state.poseIndex=Math.max(0,Math.min(index,selected.track.samples.length-1));el('poseTime').value=String(state.poseIndex);renderPose();
}
function configurePoseRange(){if(!state.poseTrack)return;const slider=el('poseTime');slider.max=String(state.poseTrack.track.samples.length-1);slider.value='0'}
function stepPose(delta){stopPose();state.poseIndex=Math.max(0,Math.min(state.poseIndex+delta,state.poseTrack.track.samples.length-1));el('poseTime').value=String(state.poseIndex);renderPose()}
function togglePose(){
  if(state.playing){stopPose();return}
  const samples=state.poseTrack.track.samples;if(state.poseIndex>=samples.length-1)state.poseIndex=0;
  state.playing=true;state.playWall=performance.now();state.playTime=samples[state.poseIndex][0];el('posePlay').textContent='暂停';requestAnimationFrame(animatePose);
}
function stopPose(){state.playing=false;el('posePlay').textContent='播放'}
function animatePose(now){
  if(!state.playing)return;
  const samples=state.poseTrack.track.samples,speed=Number(el('poseSpeed').value),target=state.playTime+(now-state.playWall)/1000*speed;
  while(state.poseIndex+1<samples.length&&samples[state.poseIndex+1][0]<=target)state.poseIndex++;
  el('poseTime').value=String(state.poseIndex);renderPose();
  if(state.poseIndex>=samples.length-1)stopPose();else requestAnimationFrame(animatePose);
}
function identity(){return{R:[1,0,0,0,1,0,0,0,1],t:[0,0,0]}}
function multiplyRotation(a,b){return[a[0]*b[0]+a[1]*b[3]+a[2]*b[6],a[0]*b[1]+a[1]*b[4]+a[2]*b[7],a[0]*b[2]+a[1]*b[5]+a[2]*b[8],a[3]*b[0]+a[4]*b[3]+a[5]*b[6],a[3]*b[1]+a[4]*b[4]+a[5]*b[7],a[3]*b[2]+a[4]*b[5]+a[5]*b[8],a[6]*b[0]+a[7]*b[3]+a[8]*b[6],a[6]*b[1]+a[7]*b[4]+a[8]*b[7],a[6]*b[2]+a[7]*b[5]+a[8]*b[8]]}
function rotate(R,v){return[R[0]*v[0]+R[1]*v[1]+R[2]*v[2],R[3]*v[0]+R[4]*v[1]+R[5]*v[2],R[6]*v[0]+R[7]*v[1]+R[8]*v[2]]}
function compose(a,b){const bt=rotate(a.R,b.t);return{R:multiplyRotation(a.R,b.R),t:[a.t[0]+bt[0],a.t[1]+bt[1],a.t[2]+bt[2]]}}
function rpyRotation(rpy){const [r,p,y]=rpy,cr=Math.cos(r),sr=Math.sin(r),cp=Math.cos(p),sp=Math.sin(p),cy=Math.cos(y),sy=Math.sin(y);return[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr,sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr,-sp,cp*sr,cp*cr]}
function axisRotation(axis,angle){const length=Math.hypot(...axis)||1,x=axis[0]/length,y=axis[1]/length,z=axis[2]/length,c=Math.cos(angle),s=Math.sin(angle),d=1-c;return[c+x*x*d,x*y*d-z*s,x*z*d+y*s,y*x*d+z*s,c+y*y*d,y*z*d-x*s,z*x*d-y*s,z*y*d+x*s,c+z*z*d]}
function transformPoint(T,p){const q=rotate(T.R,p);return[T.t[0]+q[0],T.t[1]+q[1],T.t[2]+q[2]]}
function forwardChain(chain,angles,terminalOffset){
  let T=identity();const joints=[],points=[];
  for(const joint of chain){T=compose(T,{R:rpyRotation(joint.rpy),t:joint.origin});const angle=angles[joint.name]??0;joints.push({joint,angle,point:T.t.slice()});points.push(T.t.slice());T=compose(T,{R:axisRotation(joint.axis,angle),t:[0,0,0]})}
  const endpoint=transformPoint(T,terminalOffset);points.push(endpoint);const handA=transformPoint(T,[-.005,-.038,terminalOffset[2]]),handB=transformPoint(T,[-.005,.038,terminalOffset[2]]);
  return{points,joints,endpoint,hand:[handA,handB]};
}
function project(point,W,H){const x=point[0],y=point[1],z=point[2]-.22,cy=Math.cos(state.view.yaw),sy=Math.sin(state.view.yaw),cp=Math.cos(state.view.pitch),sp=Math.sin(state.view.pitch),horizontal=cy*y+sy*x,depth=-sy*y+cy*x,vertical=cp*z-sp*depth,viewDepth=sp*z+cp*depth,scale=state.view.zoom*Math.max(.72,1+viewDepth*.34);return{x:W/2+horizontal*scale,y:H/2-vertical*scale,d:viewDepth,scale}}
function direction(a,b){const v=[b[0]-a[0],b[1]-a[1],b[2]-a[2]],length=Math.hypot(...v)||1;return v.map(value=>value/length)}
function classifyPose(chains){
  const left=direction(chains.left.points[2],chains.left.points[3]),right=direction(chains.right.points[2],chains.right.points[3]);
  const scores=[['立正',-(left[2]+right[2])/2],['侧平举',(left[1]-right[1])/2],['前平举',(left[0]+right[0])/2]].sort((a,b)=>b[1]-a[1]),best=scores[0];
  const standard=best[1]>=.72,format=v=>`${v>=0?'+':''}${v.toFixed(2)}`;
  return{label:standard?best[0]:'未形成标准动作',detail:standard?`匹配度 ${Math.round(best[1]*100)}%`:`最接近${best[0]}，匹配度仅 ${Math.round(best[1]*100)}%`,directions:`左上臂 [X ${format(left[0])}, Y ${format(left[1])}, Z ${format(left[2])}] · 右上臂 [X ${format(right[0])}, Y ${format(right[1])}, Z ${format(right[2])}]`,standard};
}
function renderPose(){
  if(!state.poseTrack)return;
  const track=state.poseTrack.track,sample=track.samples[state.poseIndex],angles=Object.fromEntries(track.names.map((name,index)=>[name,sample[2][index]]));
  const chains={};for(const side of ['left','right'])chains[side]=forwardChain(REPORT.robot_model.chains[side],angles,REPORT.robot_model.terminal_offset);
  drawRobot(chains,angles);renderReadout(chains,track,sample);el('poseTime').value=String(state.poseIndex);
}
function drawRobot(chains,angles){
  const canvas=el('robotCanvas'),rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1,W=rect.width,H=rect.height;canvas.width=Math.max(1,Math.round(W*dpr));canvas.height=Math.max(1,Math.round(H*dpr));const x=canvas.getContext('2d');x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,W,H);
  x.lineCap='round';x.lineJoin='round';
  const line=(a,b,color,width=2)=>{const p=project(a,W,H),q=project(b,W,H);x.strokeStyle=color;x.lineWidth=width*(p.scale/state.view.zoom);x.beginPath();x.moveTo(p.x,p.y);x.lineTo(q.x,q.y);x.stroke()};
  const label=(point,text,color)=>{const p=project(point,W,H);x.fillStyle=color;x.font='bold 11px system-ui';x.fillText(text,p.x+4,p.y-4)};
  const torsoCorners=[[-.07,-.135,.05],[-.07,.135,.05],[-.07,.145,.34],[-.07,-.145,.34],[.07,-.135,.05],[.07,.135,.05],[.07,.145,.34],[.07,-.145,.34]];
  x.fillStyle='#18243acc';x.strokeStyle='#60708b';x.lineWidth=1.5;x.beginPath();[4,5,6,7].forEach((index,order)=>{const p=project(torsoCorners[index],W,H);order?x.lineTo(p.x,p.y):x.moveTo(p.x,p.y)});x.closePath();x.fill();x.stroke();
  for(const [a,b] of [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]])line(torsoCorners[a],torsoCorners[b],'#60708b',1.5);
  line([0,0,.05],[0,0,.43],'#8290a8',7);line([0,-.145,.34],[0,.145,.34],'#8290a8',7);
  line([.03,0,.30],[.17,0,.30],'#f59e0b',3);label([.17,0,.30],'胸口朝前 (+X)','#fbbf24');
  const axisOrigin=[-.17,-.22,-.02];line(axisOrigin,[-.05,-.22,-.02],'#f97316',3);line(axisOrigin,[-.17,-.10,-.02],'#22d3ee',3);line(axisOrigin,[-.17,-.22,.10],'#a3e635',3);label([-.05,-.22,-.02],'X 前','#f97316');label([-.17,-.10,-.02],'Y 左','#22d3ee');label([-.17,-.22,.10],'Z 上','#a3e635');
  const head=project([0,0,.51],W,H);x.fillStyle='#263650';x.strokeStyle='#8290a8';x.lineWidth=2;x.beginPath();x.arc(head.x,head.y,Math.max(15,.047*head.scale),0,Math.PI*2);x.fill();x.stroke();
  const entries=[];for(const side of ['left','right']){const active=REPORT.robot_model.chains[side].some(joint=>Object.hasOwn(angles,joint.name)),color=active?SIDE_COLORS[side]:'#526079',chain=chains[side];for(let i=0;i<chain.points.length-1;i++)entries.push({a:chain.points[i],b:chain.points[i+1],color,width:i===2||i===4?10:8,depth:(project(chain.points[i],W,H).d+project(chain.points[i+1],W,H).d)/2});entries.push({a:chain.hand[0],b:chain.hand[1],color,width:7,depth:project(chain.endpoint,W,H).d})}
  entries.sort((a,b)=>a.depth-b.depth).forEach(item=>line(item.a,item.b,item.color,item.width));
  for(const side of ['left','right']){const active=REPORT.robot_model.chains[side].some(joint=>Object.hasOwn(angles,joint.name)),color=active?SIDE_COLORS[side]:'#526079',chain=chains[side];chain.joints.forEach(item=>{const p=project(item.point,W,H),range=item.joint.upper-item.joint.lower,near=Math.min(item.angle-item.joint.lower,item.joint.upper-item.angle)<=range*.02;x.fillStyle=near?'#fbbf24':color;x.strokeStyle='#08101d';x.lineWidth=2;x.beginPath();x.arc(p.x,p.y,near?6:4.5,0,Math.PI*2);x.fill();x.stroke()});const tip=project(chain.endpoint,W,H);x.fillStyle=color;x.font='bold 12px system-ui';x.fillText(side==='left'?'L':'R',tip.x+7,tip.y-7)}
}
function renderReadout(chains,track,sample){
  const unit=state.unit==='deg'?'°':' rad',scale=state.unit==='deg'?180/Math.PI:1,kind=LABELS[track.channel],sides=armSides(track.names),pose=classifyPose(chains),diagnostics=sample[3]||{},solverMeta=diagnostics.solver_meta||{};
  const residuals=diagnostics.final_residual_by_side||solverMeta.residual_by_side||{},residualValues=Object.values(residuals),errorLine=residualValues.length?`目标→最终 FK 误差：上臂 ${fmt(residualValues.reduce((sum,item)=>sum+item.upper_angle_rad,0)/residualValues.length*180/Math.PI)}° · 前臂 ${fmt(residualValues.reduce((sum,item)=>sum+item.forearm_angle_rad,0)/residualValues.length*180/Math.PI)}°${diagnostics.limited||solverMeta.limited?' · 求解/安全限制已触发':''}`:'该日志未记录目标骨段误差；重新回放后可显示目标与最终 FK 的偏差。';
  const nearLimits=Object.values(chains).flatMap(chain=>chain.joints).filter(item=>{const range=item.joint.upper-item.joint.lower;return Math.min(item.angle-item.joint.lower,item.joint.upper-item.angle)<=range*.02}).length;
  el('poseStatus').innerHTML=`<b>${kind} · ${sides}</b><span>t=${fmt(sample[0])} s · frame=${sample[1]} · 样本 ${state.poseIndex+1}/${track.samples.length}</span><span class="shape ${pose.standard?'':'bad-shape'}">FK 判定：${pose.label} · ${pose.detail}</span><span>${pose.directions}</span><span>${errorLine}</span><span>${nearLimits?`当前 ${nearLimits} 个关节接近 URDF 边界，形态可能被限位压缩。`:'当前无关节贴近 URDF 边界。'}</span>`;
  el('poseClock').textContent=`${fmt(sample[0])} s · ${state.poseIndex+1}/${track.samples.length}`;
  const rows=[];for(const side of ['left','right'])for(const item of chains[side].joints){if(!track.names.includes(item.joint.name))continue;const range=item.joint.upper-item.joint.lower,near=Math.min(item.angle-item.joint.lower,item.joint.upper-item.angle)<=range*.02;rows.push(`<div class="side-${side}" title="${item.joint.name}">${item.joint.name}</div><div class="${near?'limit':''}">${fmt(item.angle*scale)}${unit}${near?' · 边界':''}</div>`)}el('jointReadout').innerHTML=rows.join('');
}
function redrawAll(){requestAnimationFrame(()=>{state.charts.forEach(draw);renderPose()})}
function filterCharts(){state.charts.forEach(chart=>chart.card.classList.toggle('hidden',!state.channels.has(chart.track.channel)));redrawAll()}
setup();
</script></body></html>'''


if __name__ == "__main__":
    raise SystemExit(main())