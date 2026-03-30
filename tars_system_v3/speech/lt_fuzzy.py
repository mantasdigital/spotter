"""
Lithuanian Fuzzy Command Matching.

Provides fuzzy string matching for Lithuanian voice commands to handle
ASR misrecognitions without hardcoding every possible variant.

Uses Levenshtein distance with diacritics normalization so that
wav2vec2/Vosk output (often missing diacritics or with wrong characters)
can still match canonical Lithuanian commands.
"""

import re
from typing import Optional, Tuple, Dict, List


# Diacritics normalization map: Lithuanian → ASCII
_LT_DIACRITICS = str.maketrans({
    'ą': 'a', 'č': 'c', 'ę': 'e', 'ė': 'e', 'į': 'i',
    'š': 's', 'ų': 'u', 'ū': 'u', 'ž': 'z',
    'Ą': 'A', 'Č': 'C', 'Ę': 'E', 'Ė': 'E', 'Į': 'I',
    'Š': 'S', 'Ų': 'U', 'Ū': 'U', 'Ž': 'Z',
})


def normalize_lt(text: str) -> str:
    """
    Normalize Lithuanian text for fuzzy matching.

    Strips diacritics, lowercases, removes extra whitespace,
    and strips punctuation.

    Args:
        text: Input text (may contain Lithuanian diacritics)

    Returns:
        Normalized ASCII lowercase text
    """
    text = text.lower().strip()
    text = text.translate(_LT_DIACRITICS)
    # Remove punctuation except spaces
    text = re.sub(r'[^\w\s]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    return text


def levenshtein(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.

    Args:
        s1: First string
        s2: Second string

    Returns:
        Edit distance (number of insertions, deletions, substitutions)
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,        # insertion
                prev_row[j + 1] + 1,    # deletion
                prev_row[j] + cost      # substitution
            ))
        prev_row = curr_row

    return prev_row[-1]


def similarity_ratio(s1: str, s2: str) -> float:
    """
    Compute similarity ratio between two strings (0.0 to 1.0).

    Args:
        s1: First string
        s2: Second string

    Returns:
        Similarity ratio (1.0 = identical, 0.0 = completely different)
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    max_len = max(len(s1), len(s2))
    distance = levenshtein(s1, s2)
    return 1.0 - (distance / max_len)


# ─── Lithuanian Command Dictionary ────────────────────────────────────
# Maps canonical Lithuanian command → (action_category, action_key)
# Each entry can have multiple phrasings (aliases)

LT_COMMANDS: Dict[str, Tuple[str, str, List[str]]] = {
    # ── Motion commands ──
    # (canonical, category, action_key, aliases)
    "pirmyn":       ("motion", "forward",     ["pirmyn", "primiyn", "prirmyn", "pirmin", "pirmn",
                                                  "vaziuok pirmyn", "i prieki", "eik pirmyn", "eik i prieki"]),
    "atgal":        ("motion", "backward",    ["atgal", "vaziuok atgal", "eik atgal", "atbuline"]),
    "kairen":       ("motion", "turn_left",   ["kairen", "i kaire", "suk i kaire", "pasuk i kaire", "suk kairen"]),
    "desinen":      ("motion", "turn_right",  ["desinen", "i desine", "suk i desine", "pasuk i desine", "suk desinen"]),
    "stok":         ("motion", "stop",        ["stok", "sustok", "stop", "stot", "nutilk"]),
    "apsisuk":      ("motion", "turn_around", ["apsisuk", "apsisuki", "apsisukti", "apsisuk ratu",
                                                  "apsizuog", "apsizuok", "apsisug", "apsisuok"]),
    "sukis":        ("motion", "spin",        ["sukis", "sukis ratu", "pasukis", "apsisuk ratu"]),

    # ── Fun actions ──
    "sok":          ("motion", "dance",       ["sok", "pasok", "susok", "sokis", "sokt"]),
    "pakratyk":     ("motion", "wiggle",      ["pakratyk", "papurtyk", "kratyktis", "pakratytis"]),

    # ── Camera/head ──
    "paziurek i kaire":  ("motion", "head_left",   ["paziurek i kaire", "galva i kaire", "ziurek i kaire"]),
    "paziurek i desine": ("motion", "head_right",  ["paziurek i desine", "galva i desine", "ziurek i desine"]),
    "paziurek tiesiai":  ("motion", "head_center", ["paziurek tiesiai", "galva tiesiai", "ziurek tiesiai", "centrauk galva"]),
    "paziurek aukstyn":  ("motion", "head_up",     ["paziurek aukstyn", "galva aukstyn", "ziurek aukstyn"]),
    "paziurek zemyn":    ("motion", "head_down",   ["paziurek zemyn", "galva zemyn", "ziurek zemyn"]),

    # ── Behaviors ──
    "klajok":       ("behavior", "start_roam",  ["klajok", "klajoti", "paklajok", "pasivaikstyk",
                                                  "iseik pasivaikscioti", "pasizvalgyti", "pasizvalgysk",
                                                  "lakstyk", "begiok", "vaziuok", "tyrinej"]),
    "baik klajoti": ("behavior", "stop_roam",   ["baik klajoti", "sustok klajoti", "nustok klajoti",
                                                  "grizk", "griztk", "grizk atgal", "neklajok"]),
    "ziurek i mane":    ("behavior", "start_stare",  ["ziurek i mane", "ziureki mane", "ziurekimone",
                                                       "zurekimone", "ziuriek i mane", "stebek mane",
                                                       "siaurekimone", "zureki mone"]),
    "sek mane":         ("behavior", "start_follow", ["sek mane", "sekmane", "sek paskui mane",
                                                       "sekpaskuimane", "eik paskui mane",
                                                       "eikpaskuimane", "ateik pas mane",
                                                       "ateikpasmane", "ateik cia", "ateikcia"]),
    "nustok sekti":     ("behavior", "stop_follow",  ["nustok sekti", "nesek manes", "stovek",
                                                       "lik cia", "likcia", "nebesek"]),

    # ── System ──
    "kalbek lietuviskai": ("system", "switch_lt",   ["kalbek lietuviskai", "pakeisk i lietuviu",
                                                      "lietuviu kalba", "lietuviskai"]),
    "kalbek angliskai":   ("system", "switch_en",   ["kalbek angliskai", "pakeisk i anglu",
                                                      "anglu kalba", "angliskai"]),
    "isvalyk atminti":    ("system", "clear_memory", ["isvalyk atminti", "istrink atminti",
                                                       "pamirsk viska", "nuvalyk atminti"]),

    # ── Vision ──
    "ka matai":     ("vision", "what_see",    ["ka matai", "kamatai", "ka tu matai",
                                                "ka dabar matai", "aprasyk ka matai"]),

    # ── Web search ──
    "paiesk":       ("web", "search",         ["paiesk", "paieskot", "susirasyk",
                                                "surask", "paieskok"]),
}


def fuzzy_match_lt_command(
    text: str,
    threshold: float = 0.65,
    min_length: int = 2
) -> Optional[Tuple[str, str, str, float]]:
    """
    Fuzzy-match text against Lithuanian command dictionary.

    Normalizes diacritics and uses Levenshtein similarity to find
    the best matching Lithuanian command.

    Args:
        text: Input text (possibly ASR output with errors)
        threshold: Minimum similarity ratio to consider a match (0.0-1.0)
        min_length: Minimum text length to attempt matching

    Returns:
        Tuple of (canonical_command, category, action_key, similarity_score)
        or None if no match above threshold
    """
    if not text or len(text.strip()) < min_length:
        return None

    normalized = normalize_lt(text)

    if not normalized:
        return None

    best_match = None
    best_score = 0.0

    for canonical, (category, action_key, aliases) in LT_COMMANDS.items():
        for alias in aliases:
            norm_alias = normalize_lt(alias)

            # Exact match (fast path)
            if normalized == norm_alias:
                return (canonical, category, action_key, 1.0)

            # Check if input contains the alias or alias contains the input
            # This handles cases like "vaziuok pirmyn 50" matching "vaziuok pirmyn"
            if norm_alias in normalized or normalized in norm_alias:
                # Score based on length overlap
                overlap = min(len(normalized), len(norm_alias)) / max(len(normalized), len(norm_alias))
                if overlap > best_score:
                    best_score = overlap
                    best_match = (canonical, category, action_key, overlap)
                continue

            # Fuzzy match (with and without spaces)
            score = similarity_ratio(normalized, norm_alias)

            # Also try space-stripped comparison — ASR often inserts
            # spaces into single words (e.g., "klo jog" for "klajok")
            norm_nospace = normalized.replace(' ', '')
            alias_nospace = norm_alias.replace(' ', '')
            score_nospace = similarity_ratio(norm_nospace, alias_nospace)
            score = max(score, score_nospace)

            if score > best_score:
                best_score = score
                best_match = (canonical, category, action_key, score)

    if best_match and best_score >= threshold:
        # For very short inputs (<= 5 chars), require higher confidence
        # to avoid false positives on short English words
        if len(normalized) <= 5 and best_score < 0.75:
            return None
        return best_match

    return None


def extract_lt_number(text: str) -> Optional[float]:
    """
    Extract a number from Lithuanian text.

    Handles both digit numbers and Lithuanian word numbers.

    Args:
        text: Input text possibly containing a number

    Returns:
        Extracted number, or None
    """
    # Try digit number first
    match = re.search(r'(\d+)', text)
    if match:
        return float(match.group(1))

    # Lithuanian word numbers
    lt_numbers = {
        'nulis': 0, 'vienas': 1, 'du': 2, 'trys': 3, 'keturi': 4,
        'penki': 5, 'sesi': 6, 'septyni': 7, 'astuoni': 8, 'devyni': 9,
        'desimt': 10, 'dvylika': 12, 'penkiolika': 15,
        'dvidesimt': 20, 'trisdesimt': 30, 'keturiasdesimt': 40,
        'penkiasdesimt': 50, 'sesiasdesimt': 60, 'septyniasdesimt': 70,
        'astuoniasdesimt': 80, 'devyniasdesimt': 90, 'simtas': 100,
    }

    normalized = normalize_lt(text)
    words = normalized.split()

    total = 0
    found = False

    for word in words:
        # Try exact match
        if word in lt_numbers:
            val = lt_numbers[word]
            found = True
            if val == 100:
                total = total * 100 if total > 0 else 100
            else:
                total += val
        else:
            # Try fuzzy match for number words
            for lt_word, val in lt_numbers.items():
                if similarity_ratio(word, lt_word) >= 0.75:
                    found = True
                    if val == 100:
                        total = total * 100 if total > 0 else 100
                    else:
                        total += val
                    break

    return total if found else None


def is_lithuanian_text(text: str) -> bool:
    """
    Check if text likely contains Lithuanian language.

    Uses both diacritics detection and common Lithuanian word patterns.

    Args:
        text: Input text

    Returns:
        True if text appears to be Lithuanian
    """
    # Check for Lithuanian diacritics
    if re.search(r'[ąčęėįšųūž]', text.lower()):
        return True

    # Check for common Lithuanian short words/particles
    lt_markers = {
        'ir', 'ar', 'tai', 'ne', 'kas', 'ka', 'kaip', 'kur', 'del',
        'man', 'tu', 'jis', 'ji', 'mes', 'jie', 'jos', 'mane', 'tave',
        'cia', 'ten', 'dabar', 'taip', 'gerai', 'aciu', 'labas',
        'pirmyn', 'atgal', 'stok', 'klajok', 'sek'
    }

    words = set(normalize_lt(text).split())
    lt_word_count = len(words & lt_markers)

    # If 2+ Lithuanian marker words, likely Lithuanian
    return lt_word_count >= 2
