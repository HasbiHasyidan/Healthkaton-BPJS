"""
Fraud Detector Use Case — Hybrid AI Engine with Explainable AI (XAI).

Orchestrates:
1. Rule-Based Engine  → Detects unbundling (procedure code manipulation)
2. AI/ML Engine       → Detects upcoding (inflated diagnosis/charges)
3. XAI Narrative      → Human-readable explanation in Bahasa Indonesia

Outputs probabilistic score + narrative per prompt.md spec.
"""

import json
import logging
from typing import List, Optional, Tuple

from app.domain.schemas import (
    ClaimAnalysisRequest,
    AnalysisResult,
    BillingCodeTranslation,
    FraudStatus,
    RiskLevel,
)
from app.use_cases.dictionary_mapper import (
    translate_codes,
    translate_code,
    translate_billing_codes_display,
)
from app.infrastructure.ml_repository import ml_repo

logger = logging.getLogger(__name__)

# ── Known Unbundling Violation Pairs ─────────────────────────
# Sets of codes that should be billed as a single bundled procedure
UNBUNDLING_SETS = [
    # Caesar + Anesthesia + CPR → should be single bundled package
    ({"O80", "03.90", "93.90"}, "Persalinan Caesar dipecah: Caesar + Anestesi Epidural + CPR ditagihkan terpisah"),
    ({"74.1", "03.90", "93.90"}, "Seksio Sesarea dipecah: operasi + Anestesi Epidural + CPR ditagihkan terpisah"),
    # PTCA + IV injection + Echo
    ({"36.01", "99.29", "88.72"}, "PTCA dipecah: angioplasti + injeksi IV + ekokardiografi ditagihkan terpisah"),
    # Appendectomy + transfusion + intubation
    ({"47.01", "99.23", "96.04"}, "Apendektomi dipecah: operasi + transfusi + intubasi ditagihkan terpisah"),
    # Hip replacement + spinal tap + electrostim
    ({"81.51", "03.09", "93.54"}, "Penggantian panggul dipecah: operasi + pungsi lumbal + elektrostimulasi terpisah"),
    # Fracture reduction + electrostim + nerve block
    ({"79.31", "93.54", "04.81"}, "Reduksi fraktur dipecah: operasi + elektrostimulasi + blok saraf ditagihkan terpisah"),
    # Cholecystectomy + IV + X-ray
    ({"51.22", "99.29", "88.01"}, "Kolesistektomi dipecah: operasi + injeksi IV + rontgen ditagihkan terpisah"),
    # Hemodialisa split
    ({"39.95", "99.21"}, "Cuci darah dipisah dari injeksi"),
    # Usus buntu split
    ({"47.09", "03.90"}, "Usus buntu dipisah dari anestesi umum"),
]

# Codes commonly associated with surgery vs support procedures
SURGERY_CODES = {"36.01", "47.01", "44.13", "79.31", "81.51", "68.49", "51.22", "85.41", "74.1"}
SUPPORT_CODES = {"99.29", "99.23", "03.09", "03.90", "04.81", "93.54", "93.90", "96.04"}

# Suspicious diagnosis-notes mismatches for upcoding detection
MILD_NOTE_KEYWORDS = [
    "ringan", "biasa", "demam", "batuk", "pilek", "flu", "ispa",
    "diare ringan", "mual", "pusing", "sakit kepala ringan", "normal", "rutin"
]

SEVERE_DIAGNOSIS_CODES = {"I21.0", "C34.9", "S72.0", "B20", "N18.6", "I50.9", "A09.9"}


class FraudDetector:
    """
    Hybrid fraud detection engine (Rule-Based + ML).

    - Rule-Based → Unbundling (procedure code manipulation)
    - ML/XGBoost → Upcoding (inflated diagnosis/charges)
    - XAI         → Indonesian narrative explanation
    """

    def __init__(self):
        self.ml_repository = ml_repo

    # ── Main Analysis Pipeline ───────────────────────────────

    def analyze_claim(
        self,
        claim_id: str,
        doctor_notes: str,
        admin_billing_code: List[str],
        length_of_stay: int,
        total_charge: float,
    ) -> AnalysisResult:
        """
        Full hybrid fraud analysis pipeline:
        1. Rule-based unbundling check
        2. ML-based upcoding prediction
        3. Heuristic cost analysis
        4. Generate Indonesian XAI narrative
        """
        reasons_id: List[str] = []  # Indonesian reasons
        detected_fraud_type: Optional[str] = None
        engine_used = "RULE_BASED"

        # Translate billing codes for response
        billing_translated = translate_codes(admin_billing_code)

        # ── Step 1: Rule-Based Unbundling Detection ──────────
        unbundling_flagged, unbundling_reasons = self._check_unbundling(admin_billing_code)
        if unbundling_flagged:
            reasons_id.extend(unbundling_reasons)
            detected_fraud_type = "unbundling"

        # ── Step 2: ML-Based Upcoding Detection ─────────────
        ml_probability = 0.0
        if self.ml_repository.is_ready:
            engine_used = "HYBRID"
            ml_probability = self.ml_repository.predict_fraud_probability({
                "length_of_stay": length_of_stay,
                "total_cost": total_charge,
                "diagnosis_code": admin_billing_code[0] if admin_billing_code else "UNKNOWN",
                "procedure_codes": admin_billing_code,
            })
        else:
            # Fallback to heuristic
            ml_probability = self._heuristic_upcoding_score(
                doctor_notes, admin_billing_code, total_charge, length_of_stay
            )

        # ── Step 3: Notes vs Billing Mismatch Detection ─────
        mismatch_flagged, mismatch_reasons = self._check_notes_billing_mismatch(
            doctor_notes, admin_billing_code, total_charge, length_of_stay
        )
        if mismatch_flagged:
            reasons_id.extend(mismatch_reasons)
            if detected_fraud_type is None:
                detected_fraud_type = "upcoding"

        # ── Step 4: Cost Anomaly Detection ──────────────────
        cost_flagged, cost_reasons = self._check_cost_anomaly(total_charge, length_of_stay)
        if cost_flagged:
            reasons_id.extend(cost_reasons)
            if detected_fraud_type is None:
                detected_fraud_type = "upcoding"

        # ── Combine Scores ──────────────────────────────────
        final_probability = ml_probability
        if unbundling_flagged:
            final_probability = max(final_probability, 0.92)
        if mismatch_flagged:
            final_probability = max(final_probability, 0.85)

        final_probability = min(final_probability, 1.0)
        risk_level = self._classify_risk(final_probability)
        status = FraudStatus.FLAGGED if final_probability >= 0.5 else FraudStatus.SAFE

        if status == FraudStatus.FLAGGED and detected_fraud_type is None:
            detected_fraud_type = "upcoding"

        # ── Generate XAI Narrative ──────────────────────────
        if status == FraudStatus.SAFE:
            narrative = self._generate_safe_narrative(doctor_notes, admin_billing_code, total_charge)
        else:
            narrative = self._generate_fraud_narrative(
                doctor_notes, admin_billing_code, total_charge,
                length_of_stay, final_probability, detected_fraud_type, reasons_id
            )

        return AnalysisResult(
            claim_id=claim_id,
            doctor_notes=doctor_notes,
            admin_billing_code=admin_billing_code,
            admin_billing_translated=[
                BillingCodeTranslation(**t) for t in billing_translated
            ],
            total_charge=total_charge,
            length_of_stay=length_of_stay,
            status=status,
            probability_score=round(final_probability, 4),
            risk_level=risk_level,
            fraud_type=detected_fraud_type,
            reason_narrative=narrative,
            engine_used=engine_used,
        )

    # ── Unbundling Detection (Rule-Based) ────────────────────

    def _check_unbundling(self, codes: List[str]) -> Tuple[bool, List[str]]:
        """Check if billing codes contain known unbundling violations."""
        reasons: List[str] = []
        codes_set = set(codes)

        # Check known violation sets
        for violation_set, description in UNBUNDLING_SETS:
            if violation_set.issubset(codes_set):
                reasons.append(f"Pelanggaran unbundling terdeteksi: {description}")

        # Generic pattern: surgery + separate support billing
        surgery_found = codes_set & SURGERY_CODES
        support_found = codes_set & SUPPORT_CODES
        if surgery_found and support_found and len(codes) > 2:
            surgery_names = [translate_code(c) for c in surgery_found]
            support_names = [translate_code(c) for c in support_found]
            reasons.append(
                f"Pola unbundling mencurigakan: {len(surgery_found)} kode bedah "
                f"({', '.join(surgery_names)}) ditagihkan terpisah dengan "
                f"{len(support_found)} kode pendukung ({', '.join(support_names)})"
            )

        # Excessive code count
        if len(codes) > 5:
            reasons.append(
                f"Jumlah kode prosedur tidak wajar ({len(codes)} kode) — "
                f"kemungkinan unbundling atau duplikasi tagihan"
            )

        return len(reasons) > 0, reasons

    # ── Notes-Billing Mismatch Detection ─────────────────────

    def _check_notes_billing_mismatch(
        self, notes: str, codes: List[str], charge: float, los: int
    ) -> Tuple[bool, List[str]]:
        """Detect mismatch between doctor's notes and admin billing codes."""
        reasons: List[str] = []
        notes_lower = notes.lower()

        # Check if mild notes are paired with severe billing codes
        has_mild_keywords = any(kw in notes_lower for kw in MILD_NOTE_KEYWORDS)
        has_severe_codes = any(c in SEVERE_DIAGNOSIS_CODES for c in codes)

        if has_mild_keywords and has_severe_codes:
            severe_names = [
                f"{c} ({translate_code(c)})"
                for c in codes if c in SEVERE_DIAGNOSIS_CODES
            ]
            reasons.append(
                f"Terjadi ketidaksesuaian antara catatan dokter dan tagihan admin. "
                f"Dokter mencatat '{notes}', namun admin menagihkan diagnosis berat: "
                f"{', '.join(severe_names)}"
            )

        # High charge for mild condition
        if has_mild_keywords and charge > 15_000_000:
            reasons.append(
                f"Biaya tidak wajar: catatan dokter menunjukkan kondisi ringan "
                f"('{notes}'), namun total tagihan Rp {charge:,.0f} sangat tinggi"
            )

        return len(reasons) > 0, reasons

    # ── Cost Anomaly Detection ───────────────────────────────

    def _check_cost_anomaly(self, charge: float, los: int) -> Tuple[bool, List[str]]:
        """Detect cost anomalies based on charge/LOS ratio."""
        reasons: List[str] = []

        if los > 0:
            cost_per_day = charge / los
            if cost_per_day > 20_000_000:
                reasons.append(
                    f"Biaya per hari (Rp {cost_per_day:,.0f}/hari) melebihi ambang batas wajar "
                    f"Rp 20.000.000/hari untuk rawat inap {los} hari"
                )
        elif los == 0 and charge > 10_000_000:
            reasons.append(
                f"Tagihan Rp {charge:,.0f} untuk rawat jalan (0 hari) tidak wajar"
            )

        if charge > 100_000_000:
            reasons.append(
                f"Total tagihan Rp {charge:,.0f} melebihi batas kewajaran umum Rp 100.000.000"
            )

        return len(reasons) > 0, reasons

    # ── Heuristic Upcoding Score ─────────────────────────────

    def _heuristic_upcoding_score(
        self, notes: str, codes: List[str], charge: float, los: int
    ) -> float:
        """Heuristic scoring when ML model is unavailable."""
        score = 0.0
        notes_lower = notes.lower()

        # Mild notes + severe codes
        has_mild = any(kw in notes_lower for kw in MILD_NOTE_KEYWORDS)
        has_severe = any(c in SEVERE_DIAGNOSIS_CODES for c in codes)
        if has_mild and has_severe:
            score += 0.45

        # High charge
        if charge > 50_000_000:
            score += 0.25
        elif charge > 20_000_000:
            score += 0.1

        # High cost/day ratio
        if los > 0:
            cost_per_day = charge / los
            if cost_per_day > 20_000_000:
                score += 0.2

        # Many codes
        if len(codes) > 4:
            score += 0.1

        return min(score, 1.0)

    # ── XAI Narrative Generators ─────────────────────────────

    def _generate_fraud_narrative(
        self,
        notes: str,
        codes: List[str],
        charge: float,
        los: int,
        probability: float,
        fraud_type: Optional[str],
        reasons: List[str],
    ) -> str:
        """Generate a human-readable fraud narrative in Bahasa Indonesia."""
        prob_pct = round(probability * 100)

        if fraud_type == "upcoding":
            return f"Upcoding: Dokter mencatat penyakit ringan, namun tagihan menggunakan kode diagnosis berat. **[Probabilitas: {prob_pct}%]**"
        elif fraud_type == "unbundling":
            return "Unbundling: Tindakan Anestesi tidak boleh ditagihkan terpisah dari Bedah Utama."
        else:
            return f"Anomali terdeteksi pada klaim ini. **[Probabilitas: {prob_pct}%]**"

    def _generate_safe_narrative(
        self, notes: str, codes: List[str], charge: float
    ) -> str:
        """Generate a safe claim narrative in Bahasa Indonesia."""
        return "Valid: Keselarasan penuh antara rekam medis dan kode tindakan."

    # ── Utility ──────────────────────────────────────────────

    @staticmethod
    def _classify_risk(probability: float) -> RiskLevel:
        if probability >= 0.7:
            return RiskLevel.HIGH
        elif probability >= 0.4:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
