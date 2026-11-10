#!/usr/bin/env python3
"""FoldBoost figures (README and blog), drawn as HTML/SVG and screenshotted at 2x.

    python tools/figures/make_figures.py                 # writes build/figures/*.html and renders docs/img/*.png
    python tools/figures/make_figures.py --html-only     # HTML only (no Node/Chrome needed)

Rendering needs Node and Playwright (PLAYWRIGHT=/path/to/node_modules/playwright if it is not globally resolvable;
FIG_CHROME=chrome to use the installed Chrome). Every number comes from docs/data/figure-data.json, which
tools/export_figure_data.py derives from the archived analyses.
Colours: one entity per slot, validated on the dark surface #0B0D12 with the dataviz validator
(all pairs: lightness band, chroma floor, CVD dE 8.4, normal-vision dE 19.1, contrast >= 3:1):
vanilla #3D74B0, Anthropic #D76549, AI3/FoldBoost #2CAB88. Brand mint #5EE6C5 is used for text accents only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
D = json.loads((REPO / "docs/data/figure-data.json").read_text())
OUT = REPO / "build" / "figures"
IMG = REPO / "docs" / "img"

C_V, C_A, C_F = "#3D74B0", "#D76549", "#2CAB88"
MINT = "#5EE6C5"
INK, INK2, MUTED, FAINT, GRID = "#F3F5F8", "#B6BDCA", "#8C94A4", "#646C7C", "rgba(255,255,255,.07)"

TOOLS = {"openfold3_ob0": "OpenFold3-ob0", "openfold3": "OpenFold3", "protenix_v1": "Protenix v1", "boltz2": "Boltz-2",
         "proteinmpnn": "ProteinMPNN", "genie3": "Genie3", "esm_if1": "ESM-IF1", "atlasfold": "AtlasFold"}

T = {
    "en": dict(
        v="Original release", a="Anthropic kit", f="FoldBoost", fs="FoldBoost server",
        f1_eye="Fresh process · H100 · 18 inputs, 8 models", f1_t="Faster than Anthropic’s kit on all eight models",
        f1_sub="Speed relative to the original release, same GPU and same inputs. Right column: FoldBoost over Anthropic’s kit.",
        f1_avg="All 18 inputs", f1_col="vs Anthropic", f1_ref="original = 1×",
        f1_foot="Geometric means, 5 repetitions per input. fast modes change outputs by Anthropic’s design; on exact-mode inputs only (outputs equal the original) the means are 1.80× and 1.96×. On 23 new inputs never used for tuning, FoldBoost kept 1.12× over Anthropic.",
        f2_eye="Serving · seconds per request", f2_t="Run as a server, OpenFold3 drops from 78 s to 20 s",
        f2_sub="FoldBoost keeps the kit loaded in one process and serves requests back to back. Outputs are byte-identical to a fresh run.",
        f2_a="OpenFold3 exact · B300 · 760 tokens", f2_b="AtlasFold exact · H100", f2_res="residues", sec="s",
        f2_foot="Vanilla and Anthropic: one fresh run per request. FoldBoost server: second request on. OpenFold3 FoldBoost value includes its checkpoint adapter.",
        f3_eye="Resident server · H100 · OpenFold3", f3_t="A memory leak in the kit, found and fixed",
        f3_sub="GPU memory after each request (nvidia-smi). The kit’s global caches kept old models alive; FoldBoost frees them without editing the kit.",
        f3_l="exact mode", f3_r="fast mode", f3_req="request", f3_mib="MiB",
        f3_ship="kit as shipped", f3_fp="FoldBoost", f3_up="original OpenFold3",
        f3_restart="restarted at its memory limit", f3_slope_a="+97%", f3_slope_b="+0.27%", f3_fast_a="~625 MiB / request", f3_fast_b="4.5 MiB / request",
        f3_foot="Same inputs, 30 requests, all 30 outputs byte-identical with and without the fix. OpenFold3-ob0: 0.0% growth in both modes after the fix.",
        b_eye="FoldBoost · open source", b1="OpenFold3 per request", b1s="{stock:.1f} s → {fp:.1f} s on B300", b2="faster than Anthropic’s kit", b2s="every model we measured",
        b3="GPU memory growth, 30 requests", b3s="the kit as shipped: +97%",
    ),
    "ko": dict(
        v="원본", a="Anthropic 키트", f="FoldBoost", fs="FoldBoost 서버",
        f1_eye="새 프로세스 실행 · H100 · 입력 18개, 모델 8개", f1_t="측정한 8개 모델 모두 Anthropic 키트보다 빠릅니다",
        f1_sub="같은 GPU, 같은 입력에서 원본 대비 속도. 오른쪽 열은 Anthropic 키트 대비 FoldBoost의 속도입니다.",
        f1_avg="전체 18개 입력", f1_col="Anthropic 대비", f1_ref="원본 = 1×",
        f1_foot="입력마다 5회 반복한 기하평균. fast 모드는 Anthropic 설계상 원본과 출력이 다르며, 출력이 원본과 같은 exact 입력만 보면 1.80×와 1.96×입니다. 튜닝에 쓰지 않은 새 입력 23개에서도 Anthropic 대비 1.12×를 유지했습니다.",
        f2_eye="서비스 운영 · 요청당 시간", f2_t="서버로 띄우면 OpenFold3 요청이 78초에서 20초로",
        f2_sub="FoldBoost는 키트를 한 프로세스에 올려 두고 요청을 연달아 처리합니다. 출력은 새로 실행한 결과와 바이트 단위로 같습니다.",
        f2_a="OpenFold3 exact · B300 · 760토큰", f2_b="AtlasFold exact · H100", f2_res="잔기", sec="초",
        f2_foot="원본과 Anthropic은 요청마다 새로 실행한 시간, FoldBoost 서버는 두 번째 요청부터의 시간입니다. OpenFold3의 FoldBoost 값은 체크포인트 어댑터를 포함합니다.",
        f3_eye="상주 서버 · H100 · OpenFold3", f3_t="키트 속 메모리 누수를 찾아 고쳤습니다",
        f3_sub="요청마다 잰 GPU 메모리(nvidia-smi). 키트의 전역 캐시가 지난 모델을 놓지 않았고, FoldBoost가 키트 수정 없이 풀어 줍니다.",
        f3_l="exact 모드", f3_r="fast 모드", f3_req="요청 번호", f3_mib="MiB",
        f3_ship="키트 그대로", f3_fp="FoldBoost", f3_up="원본 OpenFold3",
        f3_restart="메모리 한도로 서버 재시작", f3_slope_a="+97%", f3_slope_b="+0.27%", f3_fast_a="요청당 약 625MiB", f3_fast_b="요청당 4.5MiB",
        f3_foot="같은 입력으로 30요청. 수정 전후 출력 30개가 모두 바이트 단위로 같습니다. OpenFold3-ob0도 수정 뒤 두 모드 모두 증가 0.0%입니다.",
        b_eye="FoldBoost · 오픈소스", b1="OpenFold3 요청당 속도", b1s="B300에서 {stock:.1f}초 → {fp:.1f}초", b2="Anthropic 키트보다 빠른 모델", b2s="측정한 모델 전부",
        b3="30요청 동안 GPU 메모리 증가", b3s="키트 그대로는 +97%",
    ),
}

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=Inter+Tight:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=block" rel="stylesheet">')

BASE_CSS = f"""
*{{margin:0;padding:0;box-sizing:border-box}}
html,body{{background:#07080B}}
body{{font-family:'Inter Tight','Pretendard',-apple-system,sans-serif;color:{INK};-webkit-font-smoothing:antialiased}}
.fig{{width:1000px;padding:44px 48px 36px;background:radial-gradient(120% 90% at 0% 0%,rgba(94,230,197,.075),transparent 55%),radial-gradient(90% 80% at 100% 100%,rgba(124,140,255,.05),transparent 60%),#0B0D12;position:relative}}
.eye{{font-family:'IBM Plex Mono','Pretendard',monospace;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:{MINT};margin-bottom:14px}}
.ko .eye{{letter-spacing:.08em}}
h1{{font-size:29px;font-weight:500;letter-spacing:-.022em;line-height:1.2;color:#fff;margin-bottom:10px}}
.ko h1{{letter-spacing:-.03em;font-weight:600}}
.sub{{font-size:15px;line-height:1.55;color:{MUTED};max-width:880px}}
.legend{{display:flex;gap:26px;margin:24px 0 6px;font-size:13.5px;color:{INK2}}}
.legend span{{display:inline-flex;align-items:center;gap:9px}}
.legend i{{display:inline-block;width:12px;height:12px;border-radius:3px}}
.legend i.dash{{width:18px;height:0;border-top:2px dashed {C_V};border-radius:0}}
.legend i.dot{{border-radius:50%}}
.foot{{font-size:12.5px;line-height:1.6;color:{FAINT};margin-top:18px;max-width:880px}}
svg text{{font-family:'Inter Tight','Pretendard',sans-serif}}
.mono{{font-family:'IBM Plex Mono','Pretendard',monospace}}
"""


def page(lang, body, extra_css=""):
    return (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">{FONTS}'
            f'<style>{BASE_CSS}{extra_css}</style></head><body class="{lang}">{body}</body></html>')


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------- figure 1: dumbbell, speed vs vanilla
def fig_speed(lang):
    L = T[lang]
    g = D["geo"]
    rows = [dict(name=L["f1_avg"], mode="", a=g["anthropic_speed"], f=g["ours_speed"], avg=True)]
    assert len({r["kit"] for r in D["fresh"]}) == 8 and all(r["ours_over_stock_time"] < r["anthropic_over_stock_time"] for r in D["fresh"]), "the title says all eight models"
    for r in D["fresh"]:
        rows.append(dict(name=TOOLS[r["kit"]], mode=r["mode"], a=1 / r["anthropic_over_stock_time"], f=1 / r["ours_over_stock_time"], avg=False))
    W, lab, x0, x1, colx = 904, 0, 214, 742, 904
    lo, hi = 0.5, 4.0
    X = lambda v: x0 + (v - lo) / (hi - lo) * (x1 - x0)
    top, rh = 34, 46
    H = top + rh * len(rows) + 22 + 36
    s = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">',
         '<defs><filter id="glow" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="3.2" result="b"/>'
         '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>']
    yb = top + rh * len(rows) + 22
    for v in (1, 2, 3, 4):
        s.append(f'<line x1="{X(v):.1f}" y1="{top - 8}" x2="{X(v):.1f}" y2="{yb - 8}" stroke="{GRID}" stroke-width="1"/>')
        s.append(f'<text x="{X(v):.1f}" y="{yb + 12}" font-size="13" fill="{FAINT}" text-anchor="middle" class="mono">{v}×</text>')
    s.append(f'<line x1="{X(1):.1f}" y1="{top - 8}" x2="{X(1):.1f}" y2="{yb - 8}" stroke="{C_V}" stroke-width="2" stroke-dasharray="5 5"/>')
    s.append(f'<text x="{X(1) + 8:.1f}" y="{top - 14}" font-size="12.5" fill="{INK2}">{esc(L["f1_ref"])}</text>')
    s.append(f'<text x="{colx}" y="{top - 14}" font-size="12" fill="{MUTED}" text-anchor="end" class="mono" letter-spacing=".04em">{esc(L["f1_col"])}</text>')
    for i, r in enumerate(rows):
        cy = top + rh * i + rh / 2
        if r["avg"]:
            s.append(f'<rect x="-6" y="{cy - rh / 2 + 3}" width="{W + 12}" height="{rh - 6}" rx="9" fill="rgba(94,230,197,.06)" stroke="rgba(94,230,197,.18)"/>')
        if i == 1:
            s.append(f'<line x1="0" y1="{cy - rh / 2 - 1}" x2="{W}" y2="{cy - rh / 2 - 1}" stroke="rgba(255,255,255,.06)"/>')
        wt = 600 if r["avg"] else 500
        s.append(f'<text x="8" y="{cy + 5}" font-size="15.5" font-weight="{wt}" fill="{"#fff" if r["avg"] else "#DEE3EB"}">{esc(r["name"])}'
                 + (f'<tspan dx="8" font-size="12" fill="{FAINT}" class="mono" font-weight="400">{r["mode"]}</tspan>' if r["mode"] else "") + '</text>')
        xa, xf = X(r["a"]), X(r["f"])
        s.append(f'<line x1="{xa:.1f}" y1="{cy}" x2="{xf:.1f}" y2="{cy}" stroke="rgba(44,171,136,.55)" stroke-width="3" stroke-linecap="round"/>')
        s.append(f'<circle cx="{xa:.1f}" cy="{cy}" r="6.5" fill="{C_A}" stroke="#0B0D12" stroke-width="2"/>')
        s.append(f'<circle cx="{xf:.1f}" cy="{cy}" r="7.5" fill="{C_F}" stroke="#0B0D12" stroke-width="2" filter="url(#glow)"/>')
        # value labels: Anthropic to the left of its dot when it sits left of FoldBoost
        s.append(f'<text x="{xa - 12:.1f}" y="{cy + 4.5}" font-size="13" fill="{MUTED}" text-anchor="end">{r["a"]:.2f}×</text>')
        s.append(f'<text x="{xf + 14:.1f}" y="{cy + 5}" font-size="15" font-weight="600" fill="#fff">{r["f"]:.2f}×</text>')
        s.append(f'<text x="{colx}" y="{cy + 5}" font-size="{16 if r["avg"] else 15}" font-weight="600" fill="{MINT}" text-anchor="end" class="mono">×{r["f"] / r["a"]:.2f}</text>')
    s.append("</svg>")
    body = (f'<div class="fig" id="fig"><div class="eye">{esc(L["f1_eye"])}</div><h1>{esc(L["f1_t"])}</h1><div class="sub">{esc(L["f1_sub"])}</div>'
            f'<div class="legend"><span><i class="dash"></i>{esc(L["v"])}</span><span><i class="dot" style="background:{C_A}"></i>{esc(L["a"])}</span>'
            f'<span><i class="dot" style="background:{C_F}"></i>{esc(L["f"])}</span></div>{"".join(s)}<div class="foot">{esc(L["f1_foot"])}</div></div>')
    return page(lang, body)


# ---------------------------------------------------------------- figure 2: seconds per request, serving
def bars(L, groups, scale_max, ticks, x0, x1, bh, gap, ggap, label_w, show_group, top=8):
    """groups: [(title or None, [(series, seconds, speed or None)])]"""
    X = lambda v: x0 + v / scale_max * (x1 - x0)
    col = {"v": C_V, "a": C_A, "f": C_F}
    name = {"v": L["v"], "a": L["a"], "f": L["fs"]}
    y = top
    parts = []
    for gt, items in groups:
        if gt:
            parts.append(f'<text x="0" y="{y + bh * 1.5 + gap + 6}" font-size="14.5" font-weight="500" fill="#DEE3EB">{esc(gt)}</text>')
        for k, (ser, secs, spd) in enumerate(items):
            w = X(secs) - x0
            parts.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="{bh}" rx="4" fill="{col[ser]}"' + (' filter="url(#glow2)"' if ser == "f" else "") + '/>')
            if show_group:
                parts.append(f'<text x="{x0 - 14}" y="{y + bh / 2 + 5}" font-size="14" fill="{INK2}" text-anchor="end">{esc(name[ser])}</text>')
            unit = L["sec"]
            val = f"{secs:.1f} s" if unit == "s" else f"{secs:.1f}{unit}"
            sp = f'<tspan dx="10" fill="{MINT if ser == "f" else MUTED}" font-weight="{600 if ser == "f" else 400}" class="mono" font-size="{13.5 if bh > 20 else 12}">{spd:.2f}×</tspan>' if spd else ""
            parts.append(f'<text x="{X(secs) + 12:.1f}" y="{y + bh / 2 + 5}" font-size="{16 if bh > 20 else 13}" font-weight="{600 if ser == "f" else 500}" fill="#fff">{val}{sp}</text>')
            y += bh + gap
        y += ggap
    axis_y = y - ggap + 6
    for v in ticks:
        parts.insert(0, f'<line x1="{X(v):.1f}" y1="{top - 6}" x2="{X(v):.1f}" y2="{axis_y}" stroke="{GRID}"/>')
        parts.append(f'<text x="{X(v):.1f}" y="{axis_y + 18}" font-size="12" fill="{FAINT}" text-anchor="middle" class="mono">{v}</text>')
    return parts, axis_y + 26


def fig_serve(lang):
    L = T[lang]
    of = next(c for c in D["chains"] if c["kind"] == "openfold3_b300")
    st = of["stock"]
    W = 904
    defs = ('<defs><filter id="glow2" x="-10%" y="-80%" width="120%" height="260%"><feGaussianBlur stdDeviation="4" result="b"/>'
            '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>')
    a_parts, a_h = bars(L, [(None, [("v", st, None), ("a", of["anthropic"], st / of["anthropic"]), ("f", of["resident_adapter"], st / of["resident_adapter"])])],
                        80, [0, 20, 40, 60, 80], 190, 740, 34, 12, 0, 190, True)
    svg_a = f'<svg width="{W}" height="{a_h}" xmlns="http://www.w3.org/2000/svg">{defs}{"".join(a_parts)}</svg>'
    groups = []
    for c in (c for c in D["chains"] if c["kind"] == "atlasfold"):
        s0 = c["stock"]
        groups.append((f'{c["size"]} {L["f2_res"]}', [("v", s0, None), ("a", c["anthropic"], s0 / c["anthropic"]), ("f", c["resident"], s0 / c["resident"])]))
    b_parts, b_h = bars(L, groups, 30, [0, 10, 20, 30], 190, 740, 15, 5, 18, 190, False)
    svg_b = f'<svg width="{W}" height="{b_h}" xmlns="http://www.w3.org/2000/svg">{defs}{"".join(b_parts)}</svg>'
    css = f".sec{{font-family:'IBM Plex Mono','Pretendard',monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{INK2};margin:30px 0 14px;padding-top:18px;border-top:1px solid rgba(255,255,255,.06)}}.sec.first{{border-top:0;padding-top:0;margin-top:26px}}"
    body = (f'<div class="fig" id="fig"><div class="eye">{esc(L["f2_eye"])}</div><h1>{esc(L["f2_t"])}</h1><div class="sub">{esc(L["f2_sub"])}</div>'
            f'<div class="legend"><span><i style="background:{C_V}"></i>{esc(L["v"])}</span><span><i style="background:{C_A}"></i>{esc(L["a"])}</span>'
            f'<span><i style="background:{C_F}"></i>{esc(L["fs"])}</span></div>'
            f'<div class="sec first">{esc(L["f2_a"])}</div>{svg_a}<div class="sec">{esc(L["f2_b"])}</div>{svg_b}<div class="foot">{esc(L["f2_foot"])}</div></div>')
    return page(lang, body, css)


# ---------------------------------------------------------------- figure 3: memory, two panels
def panel(L, series, ymax, yticks, w, h, notes, restart_at=None):
    l, r, t, b = 58, 150, 14, 34
    X = lambda v: l + (v - 1) / 29 * (w - l - r)
    Y = lambda v: t + (1 - v / ymax) * (h - t - b)
    p = [f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg">',
         '<defs><filter id="g3" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>']
    for v in yticks:
        p.append(f'<line x1="{l}" y1="{Y(v):.1f}" x2="{w - r + 10}" y2="{Y(v):.1f}" stroke="{GRID}"/>')
        p.append(f'<text x="{l - 10}" y="{Y(v) + 4:.1f}" font-size="11.5" fill="{FAINT}" text-anchor="end" class="mono">{v:,}</text>')
    for v in (1, 10, 20, 30):
        p.append(f'<text x="{X(v):.1f}" y="{h - 10}" font-size="11.5" fill="{FAINT}" text-anchor="middle" class="mono">{v}</text>')
    p.append(f'<text x="{w - r + 10}" y="{h - 10}" font-size="11" fill="{FAINT}" class="mono">{esc(L["f3_req"])}</text>')
    p.append(f'<text x="{l - 10}" y="{t - 2}" font-size="11" fill="{FAINT}" text-anchor="end" class="mono">{L["f3_mib"]}</text>')
    for key, col, dash, glow in series:
        pts = notes[key]["pts"]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(v):.1f}" for i, (a, v) in enumerate(pts))
        if glow:  # an SVG filter on a perfectly flat path has a zero-height box and drops the path; use an underlay instead
            p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-opacity=".22" stroke-width="9" stroke-linejoin="round" stroke-linecap="round"/>')
        p.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round"'
                 + (f' stroke-dasharray="{dash}"' if dash else "") + "/>")
        lx, ly = pts[-1]
        if key == restart_at:
            p.append(f'<g stroke="{col}" stroke-width="2.4" stroke-linecap="round"><line x1="{X(lx) - 6:.1f}" y1="{Y(ly) - 6:.1f}" x2="{X(lx) + 6:.1f}" y2="{Y(ly) + 6:.1f}"/>'
                     f'<line x1="{X(lx) - 6:.1f}" y1="{Y(ly) + 6:.1f}" x2="{X(lx) + 6:.1f}" y2="{Y(ly) - 6:.1f}"/></g>')
        else:
            p.append(f'<circle cx="{X(lx):.1f}" cy="{Y(ly):.1f}" r="4.5" fill="{col}" stroke="#0B0D12" stroke-width="2"/>')
        n = notes[key]
        tx, ty = n.get("at", (X(lx) + 12, Y(ly)))
        anchor = n.get("anchor", "start")
        p.append(f'<text x="{tx:.1f}" y="{ty + 4:.1f}" font-size="13" fill="#fff" font-weight="{600 if n.get("bold") else 500}" text-anchor="{anchor}">{esc(n["label"])}</text>')
        if n.get("sub"):
            p.append(f'<text x="{tx:.1f}" y="{ty + 22:.1f}" font-size="12.5" fill="{n.get("subcol", MUTED)}" class="mono" text-anchor="{anchor}" font-weight="500">{esc(n["sub"])}</text>')
    p.append("</svg>")
    return "".join(p), X, Y


def fig_memory(lang):
    L = T[lang]
    m = D["memory"]
    w, h = 440, 330
    # exact panel
    sA = [("upstream", C_V, "5 5", False), ("unguarded", C_A, None, False), ("guarded", C_F, None, True)]
    notes = {"unguarded": dict(pts=m["series"]["unguarded"], label=L["f3_ship"], sub=f'+{m["unguarded_growth_pct"]:.0f}%', subcol="#F0A58C", bold=False),
             "guarded": dict(pts=m["series"]["guarded"], label=L["f3_fp"], sub=f'+{m["guarded_growth_pct"]:.2f}%', subcol=MINT, bold=True),
             "upstream": dict(pts=m["series"]["upstream"], label=L["f3_up"], sub=None)}
    svgA, XA, YA = panel(L, sA, 4400, [0, 1000, 2000, 3000, 4000], w, h, notes)
    # place labels so they do not collide: FoldBoost above its flat line at the right end, upstream below its end
    notes["guarded"]["at"] = (XA(30) + 12, YA(737) - 22)
    notes["upstream"]["at"] = (XA(1), YA(661) + 26)
    notes["unguarded"]["at"] = (XA(30) + 12, YA(3957) - 8)
    svgA, _, _ = panel(L, sA, 4400, [0, 1000, 2000, 3000, 4000], w, h, notes)
    # fast panel
    sB = [("unguarded", C_A, None, False), ("released", C_F, None, True)]
    f = m["fast"]
    nB = {"unguarded": dict(pts=f["unguarded"], label=L["f3_ship"], sub=L["f3_fast_a"], subcol="#F0A58C"),
          "released": dict(pts=f["released"], label=L["f3_fp"], sub=L["f3_fast_b"], subcol=MINT, bold=True)}
    _, XB, YB = panel(L, sB, 14500, [0, 4000, 8000, 12000], w, h, nB, restart_at="unguarded")
    nB["unguarded"]["at"] = (XB(14) + 14, YB(13003) + 2)
    nB["released"]["at"] = (XB(30) + 12, YB(941) - 22)
    svgB, _, _ = panel(L, sB, 14500, [0, 4000, 8000, 12000], w, h, nB, restart_at="unguarded")
    restart = f'<div class="rs">✕ {esc(L["f3_restart"])}</div>'
    css = (f".pan{{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-top:22px}}.pt{{font-family:'IBM Plex Mono','Pretendard',monospace;font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{INK2};margin-bottom:10px}}"
           f".rs{{font-size:12px;color:{MUTED};margin:2px 0 0 58px}}")
    body = (f'<div class="fig" id="fig"><div class="eye">{esc(L["f3_eye"])}</div><h1>{esc(L["f3_t"])}</h1><div class="sub">{esc(L["f3_sub"])}</div>'
            f'<div class="legend"><span><i style="background:{C_A}"></i>{esc(L["f3_ship"])}</span><span><i style="background:{C_F}"></i>{esc(L["f3_fp"])}</span>'
            f'<span><i class="dash"></i>{esc(L["f3_up"])}</span></div>'
            f'<div class="pan"><div><div class="pt">{esc(L["f3_l"])}</div>{svgA}</div><div><div class="pt">{esc(L["f3_r"])}</div>{svgB}{restart}</div></div>'
            f'<div class="foot">{esc(L["f3_foot"])}</div></div>')
    return page(lang, body, css)


# ---------------------------------------------------------------- banner: the post's hero band (object-fit: cover, 1.6:1 to 3.4:1)
def fig_banner(lang):
    L = T[lang]
    of = next(c for c in D["chains"] if c["kind"] == "openfold3_b300")
    sp = of["stock"] / of["resident_adapter"]
    n_tools = len({r["kit"] for r in D["fresh"]})
    n_faster = len({r["kit"] for r in D["fresh"] if r["ours_over_stock_time"] < r["anthropic_over_stock_time"]})
    g = D["memory"]["guarded_growth_pct"]
    # pair-matrix motif: a lower-triangular grid of cells on both sides, fading toward the centre safe area
    import math
    cells = []
    for side in (-1, 1):
        for i in range(26):
            for j in range(i + 1):
                x = 40 + j * 22 if side < 0 else 1600 - 56 - j * 22
                y = 600 - 34 - i * 22
                dist = abs((x + 8) - 800) / 800
                v = (math.sin(i * 1.7 + j * 0.9 + side) * 0.5 + 0.5) * (math.cos(i * 0.31 - j * 0.57) * 0.5 + 0.5)
                op = max(0.0, (dist - 0.42) / 0.58) ** 1.6 * (0.08 + 0.42 * v)
                if op < 0.015:
                    continue
                hue = "94,230,197" if (i + 2 * j + (side > 0)) % 5 else "124,140,255"
                cells.append(f'<rect x="{x:.0f}" y="{y:.0f}" width="16" height="16" rx="3" fill="rgba({hue},{op:.3f})"/>')
    css = """
body{width:1600px;height:600px;overflow:hidden}
.bn{position:relative;width:1600px;height:600px;background:radial-gradient(60% 90% at 50% 120%,rgba(94,230,197,.10),transparent 60%),radial-gradient(40% 60% at 15% 10%,rgba(124,140,255,.08),transparent 70%),#07080B;overflow:hidden}
.bn svg{position:absolute;inset:0}
.core{position:absolute;left:50%;top:50%;transform:translate(-50%,-50%);width:900px;text-align:center}
.be{font-family:'IBM Plex Mono','Pretendard',monospace;font-size:15px;letter-spacing:.24em;text-transform:uppercase;color:#5EE6C5;margin-bottom:30px}
.ko .be{letter-spacing:.14em}
.stats{display:grid;grid-template-columns:repeat(3,1fr)}
.st{padding:0 18px;position:relative}
.st+.st:before{content:"";position:absolute;left:0;top:12px;bottom:8px;width:1px;background:linear-gradient(transparent,rgba(255,255,255,.16),transparent)}
.n{font-family:'Inter Tight',sans-serif;font-weight:300;font-size:98px;letter-spacing:-.045em;line-height:1;color:#fff;font-variant-numeric:tabular-nums}
.n em{font-style:normal;color:#5EE6C5;font-weight:300}
.l{font-size:19px;font-weight:500;color:#E6EAF0;margin-top:18px;letter-spacing:-.01em}
.s{font-family:'IBM Plex Mono','Pretendard',monospace;font-size:14px;color:#8C94A4;margin-top:8px;letter-spacing:.01em}
.ko .l{font-weight:600;letter-spacing:-.02em}
.ko .s{font-family:'Pretendard',sans-serif;font-size:15px}
.line{position:absolute;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,rgba(94,230,197,.25),transparent)}
"""
    body = (f'<div class="bn" id="fig"><svg width="1600" height="600" xmlns="http://www.w3.org/2000/svg">{"".join(cells)}</svg>'
            f'<div class="core"><div class="be">{esc(L["b_eye"])}</div><div class="stats">'
            f'<div class="st"><div class="n">{sp:.1f}<em>×</em></div><div class="l">{esc(L["b1"])}</div><div class="s">{esc(L["b1s"].format(stock=of["stock"], fp=of["resident_adapter"]))}</div></div>'
            f'<div class="st"><div class="n">{n_faster}<em>/</em>{n_tools}</div><div class="l">{esc(L["b2"])}</div><div class="s">{esc(L["b2s"])}</div></div>'
            f'<div class="st"><div class="n">+{g:.2f}<em>%</em></div><div class="l">{esc(L["b3"])}</div><div class="s">{esc(L["b3s"])}</div></div>'
            f'</div></div></div>')
    return page(lang, body, css)


if __name__ == "__main__":
    import subprocess
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "ko"):
        for name, fn in (("banner", fig_banner), ("speed", fig_speed), ("serve", fig_serve), ("memory", fig_memory)):
            (OUT / f"{name}-{lang}.html").write_text(fn(lang))
    print("wrote", len(list(OUT.glob("*.html"))), "html files to", OUT)
    if "--html-only" not in sys.argv:
        IMG.mkdir(parents=True, exist_ok=True)
        subprocess.run(["node", str(Path(__file__).with_name("shoot.cjs")), str(OUT), str(IMG)], check=True)
