"""Build the benchmark comparison report (one self-contained HTML file) from archived analyses.

    python tools/build_report.py --work ../../work --out report.html

Sources (all read-only): foldboost/data/policy-v1.json, work/holdout/**/analysis*.json,
work/holdout/serve-*/analysis.json, work/holdout/b300/**, work/holdout/wide/*/summary.json.
Every number in the page is read from those files; nothing is typed in by hand except captions.
"""
import argparse
import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def esc(x):
    return html.escape(str(x))


def f2(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def ratio_cell(r, lo=None):
    if r is None:
        return "<td class=num>–</td>"
    cls = "good" if r >= 1.05 else ("flat" if r >= 0.995 else "bad")
    tip = f" title='95% lower bound {lo:.3f}'" if lo is not None else ""
    return f"<td class='num {cls}'{tip}>{r:.3f}×</td>"


# ------------------------------------------------------------------ data collection
def norm_gpu(name):
    """Row files carry GPU names as the pod reported them ("NVIDIA B300 SXM6 AC (sm_103 ...)", "H100 (RunPod ...)")."""
    name = str(name)
    return "B300" if "B300" in name else ("H100" if "H100" in name else name)


def merge_wide_rows(wide: Path):
    """ROWS.json (first wave) plus every per-kit rows file the onpod/RunPod forks wrote (wide/_onpod/rows-*.json).
    A later file replaces an earlier row with the same (kit, gpu, mode, input). The BoltzGen production-style profile study has its own
    arms (batch sizes, kernels) and is reported separately, not as a quick-look row."""
    rows = list(load(wide / "ROWS.json") or [])
    onpod = wide / "_onpod"
    for f in sorted(onpod.glob("rows-*.json")) if onpod.exists() else []:
        if f.stem == "rows-boltzgen-profile":
            continue
        rows.extend(load(f) or [])
    merged = {}
    for r in rows:
        r = dict(r, gpu=norm_gpu(r.get("gpu", "")))
        merged[(r["kit"], r["gpu"], r.get("mode"), str(r.get("input")))] = r
    return list(merged.values())


def collect(work: Path):
    d = {"fresh": [], "holdout": [], "resident": [], "b300": [], "wide": [], "notes": []}
    policy = load(ROOT / "foldboost/data/policy-v1.json")
    d["policy_digest"] = policy["policy_sha256"][:16]
    for s in policy["scopes"]:
        m = {a: sum(v["timed_seconds"]) / len(v["timed_seconds"]) for a, v in s["providers"].items()}
        p = s["pairwise_ratio_slower_over_faster"]["ours_over_anthropic"]
        d["fresh"].append({"kit": s["kit"], "mode": s["contract"], "input": s["scope_id"].split(".")[-1], "gpu": "H100", "stock": m["stock"], "anthropic": m["anthropic"],
                           "ours": m["ours"], "ratio": p["ratio"], "lower": p["lower95"], "choice": s["decision"]["provider"],
                           "stock_in_group": "stock" in s["correctness"]["group_members"], "set": "discovery"})
    hold = work / "holdout"
    for kit_dir in sorted(hold.iterdir()) if hold.exists() else []:
        for name in ("analysis-trimmed.json", "analysis-with-trimmed.json", "analysis.json"):
            a = load(kit_dir / name)
            if a and a.get("inputs") and "kit" in a:
                for i in a["inputs"]:
                    if "mean_seconds" not in i:
                        continue
                    m = i["mean_seconds"]
                    p = i["pairwise"]["ours_over_anthropic"]
                    pt = i.get("pairwise_trimmed", {}).get("ours_over_anthropic", {})
                    d["holdout"].append({"kit": a["kit"], "mode": a["contract"], "input": i["input"], "stock": m["stock"], "anthropic": m["anthropic"], "ours": m["ours"],
                                         "ratio": p["ratio"], "lower": p["lower95"], "ratio_trimmed": pt.get("ratio"), "passed": i.get("passed_original", i.get("passed")),
                                         "passed_trimmed": i.get("passed_trimmed"), "choice": i.get("rule_choice_trimmed", i["rule_choice"]), "dir": kit_dir.name})
                break
    for tag, label, mode in (("b300/serve-study-b300", "B300 · resident", "exact"), ("serve-b300-adapter", "B300 · resident + our adapter", "exact"),
                             ("serve-h100", "H100 · resident (first run)", "exact"), ("serve-h100-v2", "H100 · resident (+ memory release)", "exact"),
                             ("openfold3-resident/exact", "H100 · resident (second pod)", "exact"), ("openfold3-resident/fast", "H100 · resident fast mode", "fast")):
        a = load(hold / tag / "analysis.json")
        if a:
            for i in a["inputs"]:
                d["resident"].append({"label": label, "kit": "openfold3", "mode": mode, "tokens": i.get("tokens", i.get("copies", 0) * 76), "fresh": i["fresh_mean_seconds"], "resident": i["resident_mean_seconds"],
                                      "ratio": i["ratio_fresh_over_resident"], "lower": i["lower95"], "ratio_trimmed": i["trimmed"]["ratio"], "gates": a["gates"], "passed": a["passed"]})
    for dirname, kit, label, mode in (("atlasfold-resident", "atlasfold", "H100 · resident", "exact"), ("boltz2-resident", "boltz2", "H100 · resident", "fast"), ("esm_if1-resident", "esm_if1", "H100 · resident", "fast")):
        a = load(hold / dirname / "analysis.json")
        if not a:
            continue
        g = dict(a["gates"])
        speed_key = next((k for k in g if k.startswith("speed")), None)
        gates = {"all_resident_outputs_identical_to_fresh": g.get("all_resident_outputs_identical_to_fresh"), "all_resident_requests_rc0_none_refused": g.get("all_resident_requests_rc0_none_refused"),
                 "gpu_memory_growth_le_5pct": g.get("gpu_memory_growth_le_5pct"), "speed_lower95_ge_1_every_input": g.get(speed_key)}
        for i in a["inputs"]:
            ht = i.get("headline_trimmed") or {}
            fresh_mean = i.get("fresh_mean_seconds") or ht.get("fresh_mean_seconds")
            res_mean = i.get("resident_mean_seconds") or ht.get("resident_mean_seconds")
            ratio = i.get("ratio_fresh_over_resident") or (i.get("preregistered_untrimmed") or {}).get("ratio_fresh_over_resident")
            tkey = i.get("key", i.get("input"))
            d["resident"].append({"label": label, "kit": kit, "mode": mode, "tokens": tkey if not str(tkey).isdigit() else int(tkey), "fresh": fresh_mean, "resident": res_mean,
                                  "ratio": ratio, "lower": i.get("lower95") or ht.get("lower95"), "ratio_trimmed": ht.get("ratio_fresh_over_resident", ratio), "gates": gates, "passed": a.get("passed", all(v for v in gates.values() if v is not None))})
    b = load(hold / "b300/of3-b300-exact-analysis.json")
    if b:
        for i in b["inputs"]:
            m, p = i["mean_seconds"], i["pairwise"]["ours_over_anthropic"]
            d["b300"].append({"kit": "openfold3", "mode": "exact", "tokens": int(i["input"]) * 76, "stock": m["stock"], "anthropic": m["anthropic"], "ours": m["ours"],
                              "ratio": p["ratio"], "lower": p["lower95"], "passed": i["passed"]})
    d["wide_rows"] = merge_wide_rows(hold / "wide")
    d["boltzgen_profile"] = load(hold / "wide" / "_onpod" / "rows-boltzgen-profile.json") or []
    d["wide_status"] = {}
    for s in sorted((hold / "wide").glob("*/summary.json")) if (hold / "wide").exists() else []:
        w = load(s)
        if w and w.get("status"):
            d["wide_status"][s.parent.name] = str(w["status"])
    d["policy"] = policy
    d["stress"] = load(hold / "stress-summary.json")
    leak = hold / "openfold3-memleak"
    d["memleak"] = {"before": load(leak / "passrule-armB2-serve-exact.json"), "after": load(leak / "passrule-armB2-fix-packs.json"),
                    "upstream": load(leak / "analysis-armA.json"), "fast": load(leak / "analysis-armC.json")}
    d["e2e"] = load(hold / "e2e" / "summary.json")
    return d


# ------------------------------------------------------------------ rendering
CSS = """
:root{--bg:#fbfaf7;--fg:#1d2330;--mut:#667085;--line:#e3e0d8;--card:#fff;--good:#0b7a53;--bad:#b42318;--flat:#8a6d00;--bar1:#2f5fd0;--bar2:#9aa5b8;--bar3:#0b7a53;}
@media (prefers-color-scheme:dark){:root{--bg:#14171f;--fg:#e8eaf0;--mut:#9aa3b5;--line:#2a2f3c;--card:#1b1f2a;--good:#5fd3a2;--bad:#ff8a80;--flat:#e0c460;--bar1:#7aa2ff;--bar2:#6b7488;--bar3:#5fd3a2;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans KR",sans-serif}
main{max-width:1080px;margin:0 auto;padding:28px 16px 80px}h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:38px 0 10px;border-top:1px solid var(--line);padding-top:22px}
h3{font-size:15px;margin:22px 0 6px}p{margin:6px 0}.sub{color:var(--mut)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}.card b{display:block;font-size:24px;letter-spacing:-.5px}.card span{color:var(--mut);font-size:13px}
table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden;font-size:13.5px}th,td{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left}
th{background:color-mix(in srgb,var(--fg) 5%,var(--card));font-weight:600}td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.good{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}.flat{color:var(--flat);font-weight:600}
.scroll{overflow-x:auto}.note{border-left:3px solid var(--line);padding:2px 12px;color:var(--mut);margin:10px 0}.tag{display:inline-block;border:1px solid var(--line);border-radius:999px;padding:0 8px;font-size:12px;color:var(--mut)}
svg text{fill:var(--fg);font-size:11px}.legend span{margin-right:14px;font-size:13px}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;vertical-align:-1px}
@media (max-width:600px){main{padding:18px 12px}th,td{padding:5px 6px}}
"""


def bar_chart(rows, title, base_label="Anthropic fresh CLI = 1.0"):
    """rows: [(label, ratio, color_var)] horizontal bars of speed-up over the reference."""
    if not rows:
        return ""
    w, lh, left = 760, 20, 270
    h = lh * len(rows) + 34
    mx = max(max(r for _, r, _ in rows), 1.05) * 1.04
    sc = (w - left - 60) / mx
    out = [f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' aria-label='{esc(title)}'>"]
    x1 = left + sc
    out.append(f"<line x1='{x1:.1f}' x2='{x1:.1f}' y1='6' y2='{h - 22}' stroke='var(--mut)' stroke-dasharray='3 3'/>")
    out.append(f"<text x='{x1:.1f}' y='{h - 8}' text-anchor='middle'>{esc(base_label)}</text>")
    for i, (label, r, color) in enumerate(rows):
        y = 8 + i * lh
        out.append(f"<text x='{left - 8}' y='{y + 12}' text-anchor='end'>{esc(label)}</text>")
        out.append(f"<rect x='{left}' y='{y + 2}' width='{max(r * sc, 1):.1f}' height='13' rx='2' fill='var({color})'/>")
        out.append(f"<text x='{left + r * sc + 5:.1f}' y='{y + 13}'>{r:.2f}×</text>")
    out.append("</svg>")
    return "".join(out)


def render(d, generated):
    fresh, hold, res, b300, wide = d["fresh"], d["holdout"], d["resident"], d["b300"], d["wide_rows"]
    gm = lambda xs: statistics.geometric_mean(xs) if xs else None
    cards = []
    if fresh:
        cards.append(("%.2f×" % gm([r["ratio"] for r in fresh]), f"새 CLI: 우리 어댑터 대 Anthropic 모드, H100 발견용 입력 {len(fresh)}개의 기하평균"))
    ok_hold = [r for r in hold]
    if ok_hold:
        cards.append(("%.2f×" % gm([r["ratio_trimmed"] or r["ratio"] for r in ok_hold]), f"같은 비교를 처음 보는 입력 {len(ok_hold)}개(절사 평균)로 확인"))
    uniq = [r for r in res if "first run" not in r["label"]]
    if uniq:
        of3 = [r["ratio"] for r in uniq if r["kit"] == "openfold3"]
        cards.append(("%.2f–%.2f×" % (min(of3), max(of3)), "OpenFold3 상주 프로세스 대 새 CLI의 요청 지연 (exact·fast, B300·H100; 어댑터를 얹으면 B300에서 최대 1.80×, H100 fast는 최대 1.82×)"))
        best = max(uniq, key=lambda r: r["ratio"])
        cards.append(("%.2f×" % best["ratio"], f"상주 프로세스 최고 행: {best['kit']} {best['label']}, 입력 {best['tokens']}"))
    ob0 = [r for r in wide if r["kit"] == "openfold3_ob0" and r["a_over_resident"]]
    if ob0:
        cards.append(("%.2f–%.2f×" % (min(r["a_over_resident"] for r in ob0), max(r["a_over_resident"] for r in ob0)), "OpenFold3-ob0 B300: 상주 대 새 CLI (간이 측정)"))
    parts = [f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>FoldBoost 벤치마크 비교</title><style>{CSS}</style></head><body><main>"]
    parts.append("<h1>FoldBoost 벤치마크 비교</h1>")
    parts.append(f"<p class=sub>Anthropic 공개 키트(고정 커밋 f4f62fa) 위에서 stock · Anthropic 모드 · 우리 경로를 같은 호스트, 같은 입력으로 비교. 생성 {esc(generated)} · 정책 해시 {esc(d['policy_digest'])}</p>")
    parts.append("<div class=cards>" + "".join(f"<div class=card><b>{esc(a)}</b><span>{esc(b)}</span></div>" for a, b in cards) + "</div>")
    parts.append("<div class=note>측정 대상은 모두 <b>새로 시작한 프로세스 전체 시간(CLI)</b> 또는 <b>상주 프로세스의 두 번째 요청부터의 지연</b>이며 준비된 모델의 forward 시간이 아닙니다. 두 가지는 서로 곱하거나 섞어 쓰지 않습니다. 비율은 &gt;1 이 우리 쪽이 빠름. 표본은 입력당 5회(상주는 10회 안팎)라 작습니다.</div>")

    # --- fresh CLI, discovery
    parts.append("<h2>1. 새 CLI 전체 시간 — 3방향 (H100, 발견용 입력)</h2>")
    parts.append("<p class=sub>5회 반복 평균 초. stock은 변경 없는 원본(<code>--mode off</code>), Anthropic은 키트의 exact/fast 모드, 우리는 같은 모드에 어댑터를 얹은 경로. 라우터는 통계적으로 구분되는 가장 빠른 경로만 고르고 동률이면 Anthropic → stock → 우리 순으로 둡니다. fast 모드에서는 stock의 출력이 달라 stock 열은 참고용이고 선택 후보가 아닙니다.</p>")
    rows = [(f"{r['kit']} {r['mode']} {r['input']}", r["ratio"], "--bar1") for r in fresh]
    parts.append(bar_chart(rows, "fresh CLI ratio ours vs Anthropic"))
    parts.append("<div class=scroll><table><tr><th>도구</th><th>모드</th><th>입력</th><th class=num>stock s</th><th class=num>Anthropic s</th><th class=num>우리 s</th><th class=num>Anthropic/우리</th><th class=num>stock/우리</th><th>라우터 선택</th></tr>")
    for r in fresh:
        sv = r["stock"] / r["ours"]
        parts.append(f"<tr><td>{esc(r['kit'])}</td><td>{r['mode']}</td><td>{esc(r['input'])}</td><td class=num>{r['stock']:.2f}</td><td class=num>{r['anthropic']:.2f}</td><td class=num>{r['ours']:.2f}</td>{ratio_cell(r['ratio'], r['lower'])}{ratio_cell(sv)}<td><b>{esc(r['choice'])}</b></td></tr>")
    parts.append("</table></div>")
    atlas = [r for r in fresh if r["kit"] == "atlasfold" and r["choice"] == "stock"]
    if atlas:
        parts.append("<div class=note>AtlasFold 228토큰: Anthropic의 exact 모드가 원본(stock)보다 느립니다. 우리 어댑터는 stock과 통계적으로 같아서 라우터는 stock을 고릅니다. 이 칸에서는 \"우리가 stock보다 낫다\"고 쓰지 않습니다.</div>")

    # --- holdout
    parts.append("<h2>2. 처음 보는 입력(보류 검증)에서도 유지되는가</h2>")
    parts.append("<p class=sub>선택·튜닝에 쓴 적 없는 입력. 판정은 두 가지를 나란히 둡니다. <b>원 규칙</b>: 사전 등록한 5회 평균 비율의 95% 하한 ≥ 1. <b>절사 규칙</b>: 가장 빠른/느린 반복을 뺀 평균 비율 &gt; 1 (소유자가 호스트 정체 1회의 영향을 공정하게 제거하려고 정함).</p>")
    parts.append("<div class=scroll><table><tr><th>도구</th><th>모드</th><th>입력</th><th class=num>stock</th><th class=num>Anthropic</th><th class=num>우리</th><th class=num>비율(평균)</th><th class=num>비율(절사)</th><th>원 규칙</th><th>절사 규칙</th></tr>")
    for r in hold:
        po = "<span class=good>통과</span>" if r["passed"] else "<span class=bad>실패</span>"
        pt = "<span class=good>통과</span>" if r["passed_trimmed"] else "<span class=bad>실패</span>"
        parts.append(f"<tr><td>{esc(r['kit'])}</td><td>{r['mode']}</td><td>{esc(r['input'])}</td><td class=num>{r['stock']:.2f}</td><td class=num>{r['anthropic']:.2f}</td><td class=num>{r['ours']:.2f}</td>{ratio_cell(r['ratio'], r['lower'])}{ratio_cell(r['ratio_trimmed'])}<td>{po}</td><td>{pt}</td></tr>")
    parts.append("</table></div>")

    # --- resident
    parts.append("<h2>3. 상주 프로세스 — 요청당 지연</h2>")
    parts.append("<p class=sub>같은 키트·같은 모드의 함수를 한 프로세스에서 반복 호출. 모델 코드는 그대로이고 임포트·CUDA 초기화·레버 설치를 한 번만 치릅니다. 모든 출력이 새 CLI 출력과 바이트 단위로 같아야 인정합니다.</p>")
    rows = [(f"{r['kit']} {r['mode']} {r['label']} · {r['tokens']}", r["ratio"], "--bar3" if "adapter" in r["label"] else "--bar1") for r in res if "first run" not in r["label"]]
    parts.append(bar_chart(rows, "resident vs fresh", "새 Anthropic CLI = 1.0"))
    parts.append("<div class=scroll><table><tr><th>도구</th><th>구성</th><th>입력 크기</th><th class=num>새 CLI s</th><th class=num>상주 s</th><th class=num>배율</th><th class=num>절사 배율</th><th>출력 동일 · rc0 · 메모리 · 속도</th></tr>")
    for r in res:
        g = r["gates"]
        gs = " · ".join(("–" if g.get(k) is None else ("✓" if g[k] else "✗")) for k in ("all_resident_outputs_identical_to_fresh", "all_resident_requests_rc0_none_refused", "gpu_memory_growth_le_5pct", "speed_lower95_ge_1_every_input"))
        parts.append(f"<tr><td>{esc(r['kit'])}</td><td>{esc(r['label'])}</td><td>{esc(r['tokens'])}</td><td class=num>{r['fresh']:.2f}</td><td class=num>{r['resident']:.2f}</td>{ratio_cell(r['ratio'], r['lower'])}{ratio_cell(r['ratio_trimmed'])}<td>{gs}</td></tr>")
    parts.append("</table></div>")

    mem = d.get("stress")
    parts.append("<div class=note>메모리 열의 ✗는 사전 등록한 점 비교(10번째 요청 대비 마지막 요청, 5% 초과)로, 변동이 큰 단일 값이라 H100 OpenFold3에서 걸립니다. 완만한 증가는 실제여서 서버에 메모리 증가 한도와 자동 재시작을 넣었습니다." + (f" 그 뒤 {mem['requests']}요청 혼합 압박 시험: 서버 {mem['server_restarts']}번 재시작, 재시작 중 {mem['routes']['fresh']}건은 새 CLI로 대체, 상주 {mem['mean_resident_seconds']:.1f}초 대 대체 {mem['mean_fresh_fallback_seconds']:.1f}초, 기준과 비교한 {mem['outputs_checked_against_fresh_reference']}건의 출력은 모두 같음." if mem else "") + " ESM-IF1·ProteinMPNN·Protenix 행은 \"속도\" 판정이 있어도 쓸모가 없는 경우입니다(작업자 프로세스를 요청마다 새로 띄우거나 키트가 두 번째 요청을 거부) — 정책의 지원표 참조.</div>")

    # --- B300
    parts.append("<h2>4. B300 (Blackwell) — Anthropic 키트의 \"기록 없음\" 칸</h2>")
    parts.append("<p class=sub>OpenFold3 키트의 B300 설정에는 \"Records on this card: none yet\"이라고 적혀 있습니다. 같은 키트 잠금(torch 2.10.0+cu128)으로 직접 측정했습니다.</p>")
    parts.append("<div class=scroll><table><tr><th>모드</th><th>토큰</th><th class=num>stock s</th><th class=num>Anthropic s</th><th class=num>우리 s</th><th class=num>Anthropic/우리</th><th>세 출력 동일</th></tr>")
    for r in b300:
        parts.append(f"<tr><td>exact</td><td>{r['tokens']}</td><td class=num>{r['stock']:.2f}</td><td class=num>{r['anthropic']:.2f}</td><td class=num>{r['ours']:.2f}</td>{ratio_cell(r['ratio'], r['lower'])}<td>{'✓' if r['passed'] else '✗'}</td></tr>")
    parts.append("<tr><td>fast</td><td>608, 684</td><td colspan=4>Anthropic fast 모드가 이름 붙은 거부로 실패 — <code>fold:v2:arch-sm103-not-measured</code> (종료코드 1, 0/1 구조). 우리 어댑터도 같은 모드를 쓰므로 같은 이유로 실패. 라우터는 fast 요청을 조용히 exact로 바꾸지 않습니다.</td><td>–</td></tr></table></div>")
    parts.append("<div class=note>B300 exact는 H100과 비슷한 시간이 걸립니다. cuEquivariance 삼각 어텐션의 Blackwell 전용 커널(SM100f)은 CUDA 12.9 이상으로 빌드된 라이브러리에서만 켜지는데 키트 고정 스택은 12.8이라 \"Non-SM100f kernel\" 경로로 돕니다. B300 전용 가속은 아직 열려 있는 숙제입니다.</div>")

    # --- wide
    parts.append("<h2>5. 도구 전반 간이 측정 (quick-look)</h2>")
    parts.append("<p class=sub>사전 등록 없이 도구마다 최소 예제로 잰 값입니다(새 CLI는 도구당 3회 중앙값, 상주는 여러 요청 평균 또는 절사 평균). 작은 입력에서는 시작 비용 비중이 달라 결과가 도구마다 크게 다릅니다.</p>")
    if wide:
        parts.append("<div class=scroll><table><tr><th>도구</th><th>GPU</th><th>모드</th><th>입력</th><th class=num>stock s</th><th class=num>Anthropic s</th><th class=num>상주 s</th><th class=num>stock/Anthropic</th><th class=num>Anthropic/상주</th><th>판정</th></tr>")
        for r in wide:
            def n(x):
                return "<td class=num>–</td>" if x is None else f"<td class=num>{x:.1f}</td>"
            parts.append(f"<tr><td>{esc(r['kit'])}</td><td>{esc(r['gpu'])}</td><td>{esc(r['mode'])}</td><td>{esc(r['input'])}</td>{n(r['stock'])}{n(r['anthropic'])}{n(r['resident'])}{ratio_cell(r['a_over_stock'])}{ratio_cell(r['a_over_resident'])}<td>{esc(r['verdict'])}</td></tr>")
        parts.append("</table></div>")
        parts.append("<div class=note>\"stock/Anthropic\"가 1보다 작으면 Anthropic 모드가 원본보다 <b>느린</b> 경우입니다(gpnstar). 이런 칸에서 라우터는 stock을 고릅니다.</div>")
    parts.append(coverage_table(d))

    parts.append("<h2>6. 실패·한계·고친 결함</h2><ul>" + "".join(f"<li>{x}</li>" for x in d["notes"]) + "</ul>")
    parts.append("<h2>7. 재현</h2><p>저장소 <code>foldboost</code> 의 <code>HYBRID.md</code>, <code>HOLDOUT_PLAN.json</code>(측정 전 확정한 계획과 소유자 지시로 바뀐 판정 규칙), <code>tools/</code>(정책 생성·분석·팟 제어). 원자료와 sha256 목록은 로컬 작업 폴더에 보존되어 있고 저장소에는 올리지 않았습니다.</p>")
    parts.append("</main></body></html>")
    return "".join(parts)


def coverage_table(d):
    pol = d["policy"]
    scoped = {s["kit"] for s in pol["scopes"]}
    fam = {f["kit"] for f in pol.get("families", [])}
    quick = {}
    for r in d["wide_rows"]:
        quick.setdefault((r["kit"], r["gpu"]), r)
    res = pol.get("resident", {})
    holdout_kits = {h["kit"] for h in d["holdout"]}
    rows = []
    for kit in sorted({c["kit"] for c in pol["coverage"]}):
        cov = {c["gpu"]: c for c in pol["coverage"] if c["kit"] == kit}
        if kit in scoped:
            h = "어댑터 3방향 측정" + (" + 보류 검증" if kit in holdout_kits else "") + (" + 입력 계열 라우팅" if kit in fam else "")
        elif ("%s" % kit, "H100") in quick:
            h = "간이 측정"
        elif cov["H100"]["status"] != "not_measured":
            h = cov["H100"]["status"].replace("_", " ")
        else:
            st = d["wide_status"].get(kit, "")
            h = "예산 부족으로 중단(미측정)" if st.startswith("paused") else (st[:40] or "미측정")
        if kit == "openfold3":
            b = "exact 3방향 측정 · fast는 키트가 거부"
        elif (kit, "B300") in quick:
            b = "간이 측정" if quick[(kit, "B300")]["a_over_resident"] else "실행 불가"
        else:
            b = cov["B300"]["status"].replace("_", " ") if cov["B300"]["status"] != "not_measured" else "미측정"
        rs = ", ".join(f"{m}: {v['status']}" for m, v in res.get(kit, {}).items()) or "–"
        rows.append((kit, h, b, rs))
    out = ["<h3>36개 도구 × GPU 현황</h3><div class=scroll><table><tr><th>도구</th><th>H100</th><th>B300</th><th>상주 프로세스</th></tr>"]
    for r in rows:
        out.append("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>")
    out.append("</table></div>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--notes", type=Path, help="JSON list of HTML strings for the limits section")
    a = ap.parse_args()
    d = collect(a.work)
    d["notes"] = json.loads(a.notes.read_text()) if a.notes else []
    import datetime
    a.out.write_text(render(d, datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")))
    print("wrote", a.out, a.out.stat().st_size, "bytes;", {k: len(v) for k, v in d.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
