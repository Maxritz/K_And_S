"""
ECC Methodology Gates for K_And_S.

ECC is not a runtime dependency of K_and_S — K_and_S does not import ECC
packages, call ECC services, or require ECC to function. Instead, ECC's
development methodology is adopted INTO K_and_S as internal quality gates:

    plan -> test -> implement -> review -> verify -> remember

Each phase of a K_and_S porting run passes through the same discipline ECC
applies to code changes:

  - PLAN       every porting strategy must state goal, scope, and success criteria
  - TEST       verification steps exist before optimization work is trusted
  - IMPLEMENT  no placeholder\\stub outputs; real work or an honest fallback
  - REVIEW     results are re-checked against the goal before being reported
  - VERIFY     claims carry evidence labels (PASS / FAIL / NOT RUN / UNVALIDATED)
  - REMEMBER   durable learnings go to learnings\\, scratch goes to findings\\

These gates are K_and_S code. They run with or without ECC on the host. When
an ECC installation happens to be present (see ecc_bridge.py), its agent
guidance can be borrowed read-only to enrich the REVIEW phase — but that is
an opportunistic bonus, never a requirement.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Evidence labels — same vocabulary the K_and_S readiness reports already use.
# ---------------------------------------------------------------------------

EVIDENCE_LABELS = ("PASS", "FAIL", "NOT RUN", "UNVALIDATED")

@dataclass
class GateResult:
    """Outcome of one methodology gate."""
    gate: str
    label: str            # one of EVIDENCE_LABELS
    detail: str
    evidence: List[str] = field(default_factory=list)

@dataclass
class PhaseReport:
    """A porting phase audited against the methodology."""
    phase: str
    started: str
    gate_results: List[GateResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "started": self.started,
            "gates": [
                {"gate": g.gate, "label": g.label, "detail": g.detail,
                 "evidence": g.evidence}
                for g in self.gate_results
            ],
            "notes": self.notes,
            "all_pass": all(g.label == "PASS" for g in self.gate_results),
            "blockers": [g.gate for g in self.gate_results if g.label == "FAIL"],
        }


class ECCMethodologyGates:
    """
    Internal quality gates implementing the ECC loop for K_and_S porting work.

    Usage:
        gates = ECCMethodologyGates()
        report = gates.audit_plan(strategy_dict)
        report = gates.audit_implementation(result_dict)
        ...
        summary = gates.summary()   # full loop status for this session
    """

    def __init__(self) -> None:
        self.phases: Dict[str, PhaseReport] = {}
        self.history: List[Dict[str, Any]] = []
        self.learnings_dir = Path(__file__).resolve().parent.parent / "learnings"
        self.findings_dir = Path(__file__).resolve().parent.parent / "findings"

    # -- lifecycle -----------------------------------------------------------

    async def initialize(self) -> Dict[str, Any]:
        """Report gate availability. Kept async for interface compatibility."""
        return {
            "status": "initialized",
            "message": "ECC methodology gates active inside K_and_S (no ECC dependency)",
            "capabilities": {
                "plan_gate": True,
                "test_gate": True,
                "implementation_gate": True,
                "review_gate": True,
                "verification_gate": True,
                "memory_gate": True,
            },
            "evidence_labels": list(EVIDENCE_LABELS),
        }

    # -- individual gates ------------------------------------------------------

    def audit_plan(self, strategy: Dict[str, Any]) -> PhaseReport:
        """
        PLAN gate: a porting strategy must have a goal, a scope, and success
        criteria before any implementation effort is spent on it.
        """
        rep = self._new_phase("plan")
        goal = strategy.get("goal") or strategy.get("recommended_approach")
        scope = strategy.get("scope") or strategy.get("files_in_scope")
        criteria = (
            strategy.get("success_criteria")
            or strategy.get("target_tok_per_s")
            or strategy.get("estimated_performance")
        )

        rep.gate_results.append(self._gate(
            "plan.goal", "PASS" if goal else "FAIL",
            "strategy states what it is trying to achieve" if goal
            else "strategy has no goal or approach",
            [f"goal={goal!r}"] if goal else [],
        ))
        rep.gate_results.append(self._gate(
            "plan.scope", "PASS" if scope else "NOT RUN",
            "scope is bounded" if scope else "scope unconstrained — guardrails cannot be checked",
            [f"scope={scope!r}"] if scope else [],
        ))
        rep.gate_results.append(self._gate(
            "plan.success_criteria", "PASS" if criteria else "NOT RUN",
            "a measurable success criterion exists" if criteria
            else "no measurable criterion — 'done' is undefined",
            [f"criteria={criteria!r}"] if criteria else [],
        ))
        self._finish(rep)
        return rep

    def audit_tests(self, plan: Dict[str, Any]) -> PhaseReport:
        """
        TEST gate: verification steps must be planned before optimization is
        trusted. A port without a verification plan measures the wrong thing.
        """
        rep = self._new_phase("test")
        has_verify = bool(plan.get("verification_steps") or plan.get("verify_numerical_equivalence"))
        has_fallback = bool(plan.get("fallback") or plan.get("rollback_plan") or plan.get("fallback_strategy"))

        rep.gate_results.append(self._gate(
            "test.verification_planned", "PASS" if has_verify else "FAIL",
            "verification steps exist in the plan" if has_verify
            else "no verification planned — results would be untrusted",
        ))
        rep.gate_results.append(self._gate(
            "test.fallback_planned", "PASS" if has_fallback else "NOT RUN",
            "a fallback path exists for failure" if has_fallback
            else "no fallback declared — failure handling undefined",
        ))
        self._finish(rep)
        return rep

    def audit_implementation(self, result: Dict[str, Any]) -> PhaseReport:
        """
        IMPLEMENT gate: outputs must be real. Placeholder values, hardcoded
        stubs, or toy data presented as results are FAIL.
        """
        rep = self._new_phase("implement")
        status = result.get("status")
        errors = result.get("errors") or []
        metrics = result.get("metrics") or {}
        strategy = result.get("optimal_strategy_selected") or {}

        rep.gate_results.append(self._gate(
            "implement.status_real", "PASS" if status == "success" else "FAIL",
            f"run reported status={status!r}",
            [f"errors={errors[:3]}"] if errors else [],
        ))
        # A confidence of exactly 1.0 with no history is a canned value.
        conf = strategy.get("confidence")
        canned = conf is not None and conf >= 1.0
        rep.gate_results.append(self._gate(
            "implement.no_canned_values", "FAIL" if canned else "PASS",
            "confidence=1.0 with no measured basis looks canned" if canned
            else "reported values are within realistic bounds",
        ))
        # Hardcoded sentinel performance with zero timing is suspicious.
        perf = strategy.get("estimated_performance")
        total_time = metrics.get("total_time", 0)
        suspicious = perf in (85.0, 92.5, 50.0) and not total_time
        rep.gate_results.append(self._gate(
            "implement.no_sentinel_metrics", "FAIL" if suspicious else "PASS",
            "sentinel performance value with zero measured time" if suspicious
            else "metrics carry measured timing or are absent",
        ))
        self._finish(rep)
        return rep

    def audit_review(self, claims: List[str], evidence: Dict[str, Any]) -> PhaseReport:
        """
        REVIEW gate: every claim in a report must map to evidence. Claims
        without backing evidence are downgraded to UNVALIDATED.
        """
        rep = self._new_phase("review")
        unwarranted = [c for c in claims if c not in evidence]
        rep.gate_results.append(self._gate(
            "review.claims_have_evidence",
            "PASS" if not unwarranted else ("FAIL" if len(unwarranted) == len(claims) else "UNVALIDATED"),
            f"{len(claims) - len(unwarranted)}\\{len(claims)} claims backed by evidence",
            [f"unwarranted: {c}" for c in unwarranted[:5]],
        ))
        self._finish(rep)
        return rep

    def audit_verification(self, verification: Dict[str, Any]) -> PhaseReport:
        """
        VERIFY gate: correctness evidence must exist and carry an honest label.
        NOT RUN is an honest answer; a silent absence is not.
        """
        rep = self._new_phase("verify")
        label = verification.get("numerical_equivalence") or "NOT RUN"
        known = label in EVIDENCE_LABELS

        rep.gate_results.append(self._gate(
            "verify.correctness_labeled", "PASS" if known else "FAIL",
            f"numerical equivalence labeled {label!r}" if known
            else f"unknown evidence label {label!r}",
        ))
        if label == "PASS":
            rep.notes.append("correctness verified — performance tuning may proceed")
        elif label in ("FAIL",):
            rep.notes.append("correctness FAILED — do not tune performance on wrong output")
        else:
            rep.notes.append("correctness not established — treat perf numbers as unvalidated")
        self._finish(rep)
        return rep

    def audit_memory(self, artifacts: Dict[str, Any]) -> PhaseReport:
        """
        REMEMBER gate: durable knowledge goes to learnings/ (committed), raw
        run output goes to findings/ (ignored). No duplicate topic files.
        """
        rep = self._new_phase("remember")
        findings = artifacts.get("findings_written") or []
        learnings = artifacts.get("learnings_updated") or []

        rep.gate_results.append(self._gate(
            "memory.findings_are_scratch", "PASS",
            f"{len(findings)} finding file(s) recorded as ephemeral",
        ))
        if learnings:
            rep.gate_results.append(self._gate(
                "memory.learnings_revised_in_place", "PASS",
                f"{len(learnings)} learning(s) recorded",
            ))
        else:
            rep.gate_results.append(self._gate(
                "memory.nonobvious_learning", "NOT RUN",
                "nothing non-obvious was distilled this run (acceptable if true)",
            ))
        self._finish(rep)
        return rep

    # -- convenience ---------------------------------------------------------

    def run_full_loop(
        self,
        strategy: Dict[str, Any],
        plan: Dict[str, Any],
        implementation: Dict[str, Any],
        verification: Dict[str, Any],
        claims: Optional[List[str]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        artifacts: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Audit one complete porting run through all six gates."""
        self.audit_plan(strategy)
        self.audit_tests(plan)
        self.audit_implementation(implementation)
        if claims is not None and evidence is not None:
            self.audit_review(claims, evidence)
        self.audit_verification(verification)
        self.audit_memory(artifacts or {})
        return self.summary()

    def summary(self) -> Dict[str, Any]:
        """Loop status across all phases audited this session."""
        phases = {name: rep.to_dict() for name, rep in self.phases.items()}
        blockers = []
        for name, d in phases.items():
            blockers.extend(f"{name}.{b}" for b in d["blockers"])
        return {
            "methodology": "plan -> test -> implement -> review -> verify -> remember",
            "provenance": "ECC development methodology, adopted as internal K_and_S gates",
            "phases": phases,
            "blockers": blockers,
            "loop_clear": not blockers,
        }

    # -- internals -------------------------------------------------------------

    def _new_phase(self, name: str) -> PhaseReport:
        rep = PhaseReport(phase=name, started=datetime.now(timezone.utc).isoformat())
        self.phases[name] = rep
        return rep

    def _gate(self, gate: str, label: str, detail: str, evidence: Optional[List[str]] = None) -> GateResult:
        if label not in EVIDENCE_LABELS:
            raise ValueError(f"bad evidence label {label!r}")
        return GateResult(gate=gate, label=label, detail=detail, evidence=evidence or [])

    def _finish(self, rep: PhaseReport) -> None:
        self.history.append(rep.to_dict())
        self.history[:] = self.history[-20:]  # bounded session history


# ---------------------------------------------------------------------------
# Backwards-compatible alias: enhanced_server.py imports this name.
# The class keeps the async initialize() contract but is fully self-contained.
# ---------------------------------------------------------------------------

EnhancedECCIntegration = ECCMethodologyGates


if __name__ == "__main__":
    import asyncio

    async def demo():
        gates = ECCMethodologyGates()
        init = await gates.initialize()
        print(init["message"])

        gates.audit_plan({"goal": "port flash_attn", "success_criteria": ">=80 tok\\s"})
        gates.audit_tests({"verification_steps": ["verify_numerical_equivalence"], "fallback": "F32 scalar"})
        gates.audit_implementation({
            "status": "success",
            "optimal_strategy_selected": {"approach": "native_wmma", "confidence": 0.82,
                                          "estimated_performance": 87.3},
            "metrics": {"total_time": 4.2},
        })
        gates.audit_verification({"numerical_equivalence": "PASS"})
        gates.audit_memory({"findings_written": ["porting-report-x.md"], "learnings_updated": []})
        print(json.dumps(gates.summary(), indent=2))

    asyncio.run(demo())
