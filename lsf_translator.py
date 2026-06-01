"""LSF gloss → French sentence translator.

LSF (Langue des Signes Française) has its own grammar:
  - Topic first:  MOI NOM ABDELBADI  →  "Je m'appelle Abdelbadi"
  - No articles / prepositions
  - No conjugation markers

This module converts a sequence of LSF glosses into a natural French
sentence using pattern-based rules and a fallback reordering engine.
"""

import re

# ── Pattern rules ────────────────────────────────────────────────────────────
# Each rule: (regex on space-joined glosses, replacement template)
# Patterns are tried in order; first match wins.

_RULES: list[tuple[str, str]] = [
    # ── Introductions ────────────────────────────────────────────────────
    (r"^BONJOUR MOI NOM (.+?)(?:\s|$)", r"Bonjour, je m'appelle \1"),
    (r"^BONJOUR MOI APPELLE (.+?)(?:\s|$)", r"Bonjour, je m'appelle \1"),
    (r"^BONJOUR NOM (.+?)(?:\s|$)", r"Bonjour, je m'appelle \1"),
    (r"^BONJOUR MOI NOM$", r"Bonjour, je m'appelle"),
    (r"^BONJOUR MOI APPELLE$", r"Bonjour, je m'appelle"),
    (r"^BONJOUR NOM$", r"Bonjour, je m'appelle"),
    (r"^MOI NOM (.+?)(?:\s|$)", r"Je m'appelle \1"),
    (r"^MOI APPELLE (.+?)(?:\s|$)", r"Je m'appelle \1"),
    (r"^NOM (.+?)(?:\s|$)", r"Je m'appelle \1"),
    (r"^MOI NOM$", r"Je m'appelle"),
    (r"^MOI APPELLE$", r"Je m'appelle"),
    (r"^NOM$", r"Je m'appelle"),

    # ── Greetings ────────────────────────────────────────────────────────
    (r"^BONJOUR$", "Bonjour"),
    (r"^MOI BONJOUR MERCI$", "Bonjour, merci."),
    (r"^MOI BONJOUR$", "Bonjour."),
    (r"^BONJOUR MERCI$", "Bonjour, merci."),
    (r"^AU-REVOIR$", "Au revoir"),
    (r"^MERCI$", "Merci"),
    (r"^OUI$", "Oui"),
    (r"^NON$", "Non"),

    # ── Age ──────────────────────────────────────────────────────────────
    (r"BONJOUR MOI MON AGE (\d+)(?: ANS)?", r"Bonjour, j'ai \1 ans"),
    (r"BONJOUR MOI AGE (\d+)(?: ANS)?", r"Bonjour, j'ai \1 ans"),
    (r"MOI MON AGE (\d+)(?: ANS)?", r"J'ai \1 ans"),
    (r"MOI AGE (\d+)(?: ANS)?", r"J'ai \1 ans"),
    (r"MOI (\d+) ANS", r"J'ai \1 ans"),
    (r"TOI MON AGE (\d+)(?: ANS)?", r"Tu as \1 ans"),
    (r"TOI AGE (\d+)(?: ANS)?", r"Tu as \1 ans"),
    (r"TOI (\d+) ANS", r"Tu as \1 ans"),

    # ── Studies / profession ──────────────────────────────────────────────
    (r"MOI ETUDIANT (.+)", r"Je suis étudiant à \1"),
    (r"MOI ETUDIANT", "Je suis étudiant"),
    (r"MOI TRAVAILLER (.+)", r"Je travaille à \1"),
    (r"MOI TRAVAILLER", "Je travaille"),

    # ── Project ──────────────────────────────────────────────────────────
    (r"ICI PROJET FIN ETUDES", "Voici mon projet de fin d'études"),
    (r"ICI PROJET (.+)", r"Voici mon projet \1"),
    (r"ICI (.+)", r"Voici \1"),
    (r"PROJET FIN ETUDES", "Mon projet de fin d'études"),

    # ── Questions ────────────────────────────────────────────────────────
    (r"TOI NOM QUOI", "Comment tu t'appelles ?"),
    (r"TOI AGE QUOI", "Tu as quel âge ?"),
    (r"TOI VOULOIR QUOI", "Qu'est-ce que tu veux ?"),
    (r"TOI MANGER QUOI", "Qu'est-ce que tu veux manger ?"),
    (r"TOI BOIRE QUOI", "Qu'est-ce que tu veux boire ?"),
    (r"TOI HABITER QUOI", "Où est-ce que tu habites ?"),
    (r"TOI BIEN QUOI", "Comment tu vas ?"),
    (r"TOI BIEN", "Tu vas bien ?"),
    (r"QUOI TOI", "Et toi ?"),

    # ── Needs / wants ────────────────────────────────────────────────────
    (r"^MOI VOULOIR MANGER$", "Je veux manger"),
    (r"^MOI VOULOIR BOIRE$", "Je veux boire"),
    (r"^MOI VOULOIR DORMIR$", "Je veux dormir"),
    (r"^MOI VOULOIR AIDE$", "J'ai besoin d'aide"),
    (r"^MOI VOULOIR (.+)", r"Je veux \1"),
    (r"^MOI BESOIN (.+)", r"J'ai besoin de \1"),

    # ── Feelings / state ─────────────────────────────────────────────────
    (r"^MOI BIEN$", "Je vais bien"),
    (r"^MOI MAL$", "Je ne vais pas bien"),
    (r"^MOI FATIGUE$", "Je suis fatigué"),
    (r"^MOI CONTENT$", "Je suis content"),
    (r"^MOI TRISTE$", "Je suis triste"),

    # ── Actions ──────────────────────────────────────────────────────────
    (r"^MOI MANGER$", "Je mange"),
    (r"^MOI BOIRE$", "Je bois"),
    (r"^MOI DORMIR$", "Je dors"),
    (r"^MOI PARLER$", "Je parle"),
    (r"^MOI ECOUTER$", "J'écoute"),
    (r"^MOI AIMER (.+)", r"J'aime \1"),
    (r"^MOI AIMER$", "J'aime"),
    (r"^MOI COMPRENDRE$", "Je comprends"),
    (r"^MOI COMPRENDRE PAS$", "Je ne comprends pas"),
    (r"^MOI PAS COMPRENDRE$", "Je ne comprends pas"),
    (r"^MOI AIDE$", "J'ai besoin d'aide"),

    # ── Numbers standalone ───────────────────────────────────────────────
    (r"^(\d+)$", r"\1"),

    # ── Negation (PAS at end — LSF grammar) ──────────────────────────────
    (r"^MOI (.+?) PAS$", r"Je ne \1 pas"),
    (r"^TOI (.+?) PAS$", r"Tu ne \1 pas"),
]


def _apply_rules(gloss: str, *, whole: bool = False) -> str | None:
    """Try each rule against *gloss*; return the French string or None.

    When *whole* is True, only fully-anchored rules (ending with ``$``)
    that consume the entire gloss are considered. This lets fixed
    multi-pivot expressions (e.g. ``MOI BONJOUR MERCI``) be matched as a
    block, without letting open-ended rules like ``MOI ETUDIANT (.+)``
    swallow tokens that should start a new proposition (time markers…).
    """
    for pattern, replacement in _RULES:
        if whole and not pattern.endswith("$"):
            continue
        m = re.match(pattern, gloss, re.IGNORECASE)
        if m and (not whole or m.end() == len(gloss)):
            return m.expand(replacement)
    return None


# ── Segment splitter ─────────────────────────────────────────────────────────

def _split_segments(glosses: list[str]) -> list[list[str]]:
    """Split gloss list at explicit pause markers (|)."""
    segments: list[list[str]] = []
    current: list[str] = []
    for g in glosses:
        if g == "|":
            if current:
                segments.append(current)
                current = []
        else:
            current.append(g)
    if current:
        segments.append(current)
    return segments


# Words that typically start a new proposition in LSF
_PIVOT_STARTERS = {"MOI", "TOI", "LUI", "ELLE", "NOUS", "ICI",
                   "AU-REVOIR", "MERCI", "OUI", "NON", "BONJOUR",
                   # Time / context markers also start a new clause
                   "AUJOURD'HUI", "HIER", "DEMAIN", "MAINTENANT"}


def _auto_split(glosses: list[str]) -> list[list[str]]:
    """Automatically split a gloss sequence into propositions.

    A new proposition starts whenever a subject pivot (MOI, TOI, ICI…)
    appears after at least one token has already been accumulated —
    except when MOI immediately follows BONJOUR (same proposition).

    e.g. ["BONJOUR","MOI","NOM","ABDEL","MOI","mon age","24","ANS"]
      →  [["BONJOUR","MOI","NOM","ABDEL"], ["MOI","mon age","24","ANS"]]
    """
    if not glosses:
        return []

    upper = [g.upper() for g in glosses]
    propositions: list[list[str]] = []
    current: list[str] = [glosses[0]]

    for i in range(1, len(glosses)):
        g_up = upper[i]
        prev_up = upper[i - 1]

        is_pivot = g_up in _PIVOT_STARTERS
        # Keep MOI glued to BONJOUR — they form one opener
        bonjour_moi = g_up == "MOI" and prev_up == "BONJOUR"

        if is_pivot and not bonjour_moi and len(current) > 0:
            propositions.append(current)
            current = [glosses[i]]
        else:
            current.append(glosses[i])

    if current:
        propositions.append(current)

    return propositions


# ── Fallback: basic word-level mapping ───────────────────────────────────────

_WORD_MAP = {
    "MOI": "je",
    "TOI": "tu",
    "LUI": "il",
    "ELLE": "elle",
    "NOUS": "nous",
    "NOM": "m'appelle",
    "APPELLE": "m'appelle",
    "ETUDIANT": "étudiant",
    "ANS": "ans",
    "BIEN": "bien",
    "MAL": "mal",
    "OUI": "oui",
    "NON": "non",
    "MERCI": "merci",
    "BONJOUR": "bonjour",
    "AU-REVOIR": "au revoir",
    "ICI": "voici",
    "PROJET": "projet",
    "FIN": "fin",
    "ETUDES": "d'études",
    "VOULOIR": "veux",
    "AIMER": "aime",
    "MANGER": "manger",
    "BOIRE": "boire",
    "DORMIR": "dors",
    "TRAVAILLER": "travaille",
    "PARLER": "parle",
    "ECOUTER": "écoute",
    "COMPRENDRE": "comprends",
    "AIDE": "aide",
    "PAS": "pas",
    "QUOI": "quoi",
    "SUIS": "suis",
    "AI": "ai",
    "POUR": "pour",
    "CONTENT": "content",
    "TRISTE": "triste",
    "FATIGUE": "fatigué",
    "BESOIN": "besoin",
    "HABITER": "habite",
    "AGE": "âge",
    "MON AGE": "âge",
    "AUJOURD'HUI": "aujourd'hui",
    "HIER": "hier",
    "DEMAIN": "demain",
    "MAINTENANT": "maintenant",
}


def _fallback(glosses: list[str]) -> str:
    """Translate word-by-word as a last resort."""
    raw = _fallback_lower(glosses)
    return raw[0].upper() + raw[1:] if raw else ""


def _fallback_lower(glosses: list[str]) -> str:
    """Like ``_fallback`` but without forcing a leading capital.

    Used when appending leftover tokens after a rule has matched a
    prefix of the gloss sequence.
    """
    words: list[str] = []
    for g in glosses:
        mapped = _WORD_MAP.get(g.upper(), g)
        if mapped:
            words.append(mapped)
    return " ".join(words)


# ── Public API ───────────────────────────────────────────────────────────────

def _merge_digits(glosses: list[str]) -> list[str]:
    """Merge consecutive digit glosses into a single number token.

    e.g. ["MOI", "2", "4", "ANS"] → ["MOI", "24", "ANS"]
    """
    merged = []
    digit_buf = []
    for g in glosses:
        if g.isdigit():
            digit_buf.append(g)
        else:
            if digit_buf:
                merged.append("".join(digit_buf))
                digit_buf = []
            merged.append(g)
    if digit_buf:
        merged.append("".join(digit_buf))
    return merged


def _merge_letters(glosses: list[str]) -> list[str]:
    """Merge consecutive single-letter glosses into a capitalised word.

    e.g. ["NOM", "A", "B", "D", "E", "L"] → ["NOM", "Abdel"]

    This handles names spelled letter by letter without spell-mode toggle.
    A run is merged only when there are at least 2 consecutive letters.
    """
    merged = []
    letter_buf: list[str] = []

    def _flush():
        if len(letter_buf) >= 2:
            word = letter_buf[0].upper() + "".join(c.lower() for c in letter_buf[1:])
            merged.append(word)
        elif letter_buf:
            merged.append(letter_buf[0])
        letter_buf.clear()

    for g in glosses:
        if len(g) == 1 and g.isalpha():
            letter_buf.append(g)
        else:
            _flush()
            merged.append(g)
    _flush()
    return merged


def _proper_case(text: str) -> str:
    """Title-case isolated ALL-CAPS words longer than 1 char (spelled names)."""
    def _fix(m):
        w = m.group(0)
        return w.capitalize() if len(w) > 1 else w
    import re as _re
    return _re.sub(r'\b[A-Z]{2,}\b', _fix, text)


def _translate_segment(seg: list[str]) -> str:
    """Translate one proposition segment into French.

    If a rule matches only the *prefix* of the gloss sequence, the
    remaining tokens are translated with the fallback and appended,
    so that nothing the user signed gets silently dropped.
    """
    seg = _merge_digits(seg)
    seg = _merge_letters(seg)
    if not seg:
        return ""
    joined_upper = " ".join(g.upper() for g in seg)

    matched_text: str | None = None
    remainder_tokens: list[str] = []

    for pattern, replacement in _RULES:
        m = re.match(pattern, joined_upper, re.IGNORECASE)
        if m:
            matched_text = m.expand(replacement)
            consumed = joined_upper[: m.end()]
            n_consumed = len(consumed.split())
            remainder_tokens = seg[n_consumed:]
            break

    if matched_text is None:
        result = _fallback(seg)
    else:
        result = matched_text
        if remainder_tokens:
            extra = _fallback_lower(remainder_tokens)
            if extra:
                result = f"{result.rstrip(' .,;:')} {extra}"

    if result:
        result = _proper_case(result)
    return result or ""


def _dedupe_consecutive(glosses: list[str]) -> list[str]:
    """Collapse runs of identical glosses (case-insensitive).

    Avoids artefacts like ``["MOI", "MOI", "MOI"]`` produced when the
    user pauses on a sign without releasing it, which would otherwise
    yield "Je. Je. Je." in the French output.
    """
    out: list[str] = []
    last_norm: str | None = None
    for g in glosses:
        n = g.upper()
        if n != last_norm:
            out.append(g)
            last_norm = n
    return out


def translate(glosses: list[str]) -> str:
    """Translate a list of LSF glosses into a French sentence.

    Explicit pauses ("|") and subject pivots (MOI, TOI, ICI…) both
    act as proposition boundaries — no manual pause needed.
    """
    glosses = _dedupe_consecutive(glosses)
    # 1. Split at explicit pauses first
    pause_segments = _split_segments(glosses)

    french_parts: list[str] = []

    for seg in pause_segments:
        seg = _merge_digits(seg)
        seg = _merge_letters(seg)
        joined_upper = " ".join(g.upper() for g in seg)
        whole = _apply_rules(joined_upper, whole=True)
        if whole:
            french_parts.append(_proper_case(whole))
            continue

        propositions = _auto_split(seg)
        for prop in propositions:
            result = _translate_segment(prop)
            if result:
                french_parts.append(result)

    sentence = ". ".join(french_parts)
    if sentence and not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence
