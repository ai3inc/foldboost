"""Choose stock / Anthropic / ours for one request from the frozen evidence policy.

Pure functions, standard library only. The selector never measures anything and never looks at
timings of the request: it only decides whether the request lies inside a recorded evidence
scope and returns the provider the frozen rule picked for that scope. Everything else gets
Anthropic with the requested mode untouched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .requests import Request

DATA = Path(__file__).resolve().parent / "data"
DEFAULT = "anthropic"


class PolicyError(RuntimeError):
    pass


def load_policy(path: Path | None = None) -> dict:
    path = path or DATA / "policy-v1.json"
    policy = json.loads(path.read_text())
    claimed = policy.pop("policy_sha256", None)
    actual = hashlib.sha256(json.dumps(policy, indent=1, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if claimed != actual:
        raise PolicyError(f"policy digest mismatch: file says {claimed}, content is {actual}")
    policy["policy_sha256"] = actual
    return policy


@dataclass
class Facts:
    """What is known about the machine and environment the request will run in."""
    gpu_name: str | None = None
    torch: str | None = None
    triton: str | None = None
    python: str | None = None
    env: dict = field(default_factory=dict)          # process environment (only fixed variables are compared)
    satisfied: set = field(default_factory=set)       # names of preconditions that hold (e.g. warm_fingerprint_cache)
    verified_weights: set = field(default_factory=set)


@dataclass
class Decision:
    provider: str
    mode: str | None
    reason: str
    scope_id: str | None = None
    adapter: str | None = None            # adapter belonging to the matched scope (present, inert, unless enabled)
    adapter_enabled: bool = False         # True only when the chosen provider is our code
    fixed_env: dict = field(default_factory=dict)
    non_default: bool = False             # True when the choice is not the Anthropic default
    near_misses: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _fallback(request: Request, reason: str, near: list | None = None) -> Decision:
    return Decision(DEFAULT, request.mode, reason, near_misses=near or [])


def select(policy: dict, request: Request, facts: Facts) -> Decision:
    if request.problems:
        return _fallback(request, "unparsable_request: " + "; ".join(request.problems))
    if request.mode == "off":
        return Decision("stock", "off", "user_requested_stock")
    if request.mode not in ("exact", "fast"):
        return _fallback(request, f"mode_without_evidence:{request.mode}")
    if request.unrecognized:
        return _fallback(request, "unrecognized_options:" + ",".join(sorted(request.unrecognized)))
    candidates = [s for s in policy["scopes"] if s["kit"] == request.kit and s["gpu"].lower() == request.config.lower()]
    if not candidates:
        return _fallback(request, "no_evidence_for_kit_and_gpu")
    near: list = []
    for scope in candidates:
        miss = _first_mismatch(scope, request, facts)
        if miss is None:
            return _decide(scope, request)
        near.append({"scope_id": scope["scope_id"], "mismatch": miss})
    for fam in policy.get("families", []):
        if fam["kit"] != request.kit or fam["gpu"].lower() != request.config.lower():
            continue
        miss = _family_mismatch(fam, request, facts)
        if miss is None:
            return _decide_family(fam)
        near.append({"scope_id": fam["family_id"], "mismatch": miss})
    return _fallback(request, "outside_all_evidence_scopes", near)


def _family_mismatch(fam: dict, request: Request, facts: Facts) -> str | None:
    """A family covers any input of the kit inside the hull of the sizes that were tested (discovery plus unseen holdout),
    with every option fixed except the listed free ones. Same environment, weights and preconditions as an exact scope."""
    if fam["contract"] != request.mode:
        return f"mode:{request.mode}!={fam['contract']}"
    free = set(fam["free_options"])
    fixed = {k: v for k, v in request.options.items() if k not in free}
    if fixed != fam["fixed_options"]:
        keys = sorted(set(fixed) | set(fam["fixed_options"]))
        return "options:" + ",".join(k for k in keys if fixed.get(k) != fam["fixed_options"].get(k))
    for key in fam.get("require_present", []):
        if request.extra.get(key) is None:
            return f"unverified:{key}"
    lo, hi = fam["size_hull"]
    if request.size is None or not (lo <= request.size <= hi):
        return f"size:{request.size}not_in[{lo},{hi}]"
    env = fam["environment"]
    if facts.gpu_name is None or facts.gpu_name != env["gpu_name"]:
        return f"gpu_name:{facts.gpu_name!r}"
    for field_name in ("torch", "triton", "python"):
        expected = env.get(field_name)
        if expected is not None and getattr(facts, field_name) != expected:
            return f"{field_name}:{getattr(facts, field_name)!r}!={expected!r}"
    for name in fam.get("requires", []):
        if name not in facts.satisfied:
            return f"precondition:{name}"
    for weight in fam.get("weights") or []:
        if weight["sha256"] not in facts.verified_weights:
            return f"weights_unverified:{weight['name']}"
    return None


def _decide_family(fam: dict) -> Decision:
    provider = fam["decision"]["provider"]
    return Decision(
        provider=provider, mode="off" if provider == "stock" else fam["contract"], scope_id=fam["family_id"], adapter=fam["adapter"],
        adapter_enabled=provider == "ours", reason="in_family_tested_hull:" + fam["decision"]["basis"],
        fixed_env=dict(fam["environment"].get("env_fixed", {})), non_default=provider != DEFAULT,
    )


def _first_mismatch(scope: dict, request: Request, facts: Facts) -> str | None:
    if scope["contract"] != request.mode:
        return f"mode:{request.mode}!={scope['contract']}"
    if scope["match"]["input_sha256"] != request.input_sha256:
        return "input_sha256"
    if scope["match"]["options"] != request.options:
        keys = sorted(set(scope["match"]["options"]) | set(request.options))
        diff = [k for k in keys if scope["match"]["options"].get(k) != request.options.get(k)]
        return "options:" + ",".join(diff)
    env = scope["environment"]
    if facts.gpu_name is None or facts.gpu_name != env["gpu_name"]:
        return f"gpu_name:{facts.gpu_name!r}"
    for field_name in ("torch", "triton", "python"):
        expected = env.get(field_name)
        if expected is not None and getattr(facts, field_name) != expected:
            return f"{field_name}:{getattr(facts, field_name)!r}!={expected!r}"
    for name in scope.get("requires", []):
        if name not in facts.satisfied:
            return f"precondition:{name}"
    for weight in scope.get("weights") or []:
        if weight["sha256"] not in facts.verified_weights:
            return f"weights_unverified:{weight['name']}"
    return None


def _decide(scope: dict, request: Request) -> Decision:
    provider = scope["decision"]["provider"]
    mode = "off" if provider == "stock" else scope["contract"]
    return Decision(
        provider=provider, mode=mode, scope_id=scope["scope_id"], adapter=scope["adapter"],
        adapter_enabled=provider == "ours",
        reason="in_scope_frozen_rule:undominated=" + "/".join(scope["decision"]["undominated"]),
        fixed_env=dict(scope["environment"].get("env_fixed", {})), non_default=provider != DEFAULT,
    )
