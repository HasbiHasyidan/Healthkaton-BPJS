"""
Medical Dictionary Mapper — Translate ICD codes to human-readable Indonesian descriptions.

Enables verifiers to instantly understand billing codes in context.
Used by the fraud detector for XAI narratives and by the frontend for display.
"""

from typing import Dict, List, Optional

# ── ICD-10 Diagnosis Codes → Indonesian ─────────────────────

ICD10_MAP: Dict[str, str] = {
    # Infectious
    "A01.0": "Demam Tifoid (Typhoid Fever)",
    "A09.0": "Gastroenteritis Ringan",
    "A09.9": "Gastroenteritis Berat",
    "A15.0": "TB Paru (Pulmonary TB)",
    "A16.2": "TB Paru tanpa Konfirmasi",
    "B20":   "Penyakit HIV",

    # Neoplasm
    "C34.9": "Kanker Paru-Paru",

    # Blood
    "D64.9": "Anemia, tidak spesifik",

    # Endocrine
    "E11.9": "Diabetes Mellitus Tipe 2",
    "E78.5": "Hiperlipidemia",

    # Nervous
    "G40.9": "Epilepsi",

    # Circulatory
    "I10":   "Hipertensi Esensial",
    "I21.0": "Infark Miokard Akut (Serangan Jantung)",
    "I50.9": "Gagal Jantung",

    # Respiratory
    "J06.9": "Infeksi Saluran Napas Atas (ISPA)",
    "J18.9": "Pneumonia",
    "J20.9": "Bronkitis Akut",
    "J44.1": "PPOK Eksaserbasi Akut",

    # Digestive
    "K35.2": "Apendisitis Akut",
    "K80.0": "Batu Empedu (Cholelithiasis)",

    # Musculoskeletal
    "M17.1": "Osteoarthritis Lutut",
    "M54.5": "Nyeri Punggung Bawah (Low Back Pain)",

    # Genitourinary
    "N18.6": "Penyakit Ginjal Tahap Akhir (ESRD)",
    "N39.0": "Infeksi Saluran Kemih (ISK)",

    # Pregnancy
    "O80":   "Persalinan Normal",
    "O82":   "Persalinan Caesar",
    "O14.9": "Pre-eklampsia Berat",

    # Symptoms
    "R10.4": "Nyeri Perut",
    "R50.9": "Demam (Fever)",

    # Injury
    "S72.0": "Fraktur Leher Femur",

    # Health services
    "Z51.1": "Sesi Kemoterapi",
}

# ── ICD-9-CM / INA-CBG Procedure Codes → Indonesian ────────

PROCEDURE_MAP: Dict[str, str] = {
    # Surgical
    "36.01": "PTCA (Angioplasti Koroner Perkutan)",
    "47.01": "Apendektomi Laparoskopi",
    "47.09": "Bedah Usus Buntu",
    "44.13": "Gastroskopi",
    "79.31": "Reduksi Terbuka Fraktur",
    "81.51": "Penggantian Panggul Total",
    "68.49": "Histerektomi Abdominal Total",
    "51.22": "Kolesistektomi (Pengangkatan Empedu)",
    "85.41": "Mastektomi",
    "74.1":  "Seksio Sesarea (Operasi Caesar)",
    "13.41": "Operasi Katarak (Phaco)",

    # Anesthesia / Support
    "03.90": "Anestesi Umum",
    "03.09": "Spinal Tap / Pungsi Lumbal",
    "04.81": "Blok Saraf",
    "93.90": "CPR (Resusitasi Jantung Paru)",
    "93.54": "Terapi Elektrostimulasi",
    "96.04": "Intubasi Endotrakeal",
    "39.95": "Hemodialisa",

    # Transfusion / Injection
    "99.29": "Injeksi Obat Intravena (IV)",
    "99.21": "Injeksi Antibiotik",
    "99.23": "Transfusi Darah",
    "99.04": "Transfusi Trombosit",

    # Diagnostic
    "87.03": "CT Scan Kepala",
    "87.44": "Rontgen Dada (Chest X-ray)",
    "88.01": "Rontgen Abdomen",
    "88.72": "Ekokardiografi",
    "88.76": "USG Vaskular Perifer",
    "89.52": "EKG (Elektrokardiogram)",
}


def translate_code(code: str) -> str:
    """
    Translate a single ICD code to its Indonesian description.

    Falls back to the raw code if not found in the dictionary.
    """
    return ICD10_MAP.get(code, PROCEDURE_MAP.get(code, code))


def translate_codes(codes: List[str]) -> List[Dict[str, str]]:
    """
    Translate a list of ICD codes to Indonesian descriptions.

    Returns:
        List of dicts: [{"code": "R50.9", "description": "Demam (Fever)"}, ...]
    """
    return [
        {"code": code, "description": translate_code(code)}
        for code in codes
    ]


def translate_billing_codes_display(codes: List[str]) -> str:
    """
    Format billing codes into a human-readable string for dashboard display.

    Example: "R50.9 (Demam), A09.9 (Gastroenteritis Berat)"
    """
    parts = []
    for code in codes:
        desc = translate_code(code)
        if desc != code:
            parts.append(f"{code} ({desc})")
        else:
            parts.append(code)
    return ", ".join(parts)


def get_diagnosis_description(code: str) -> Optional[str]:
    """Get Indonesian description for a diagnosis code, or None."""
    return ICD10_MAP.get(code)


def get_procedure_description(code: str) -> Optional[str]:
    """Get Indonesian description for a procedure code, or None."""
    return PROCEDURE_MAP.get(code)
