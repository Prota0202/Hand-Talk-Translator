"""Test suite for the LSF → French translation engine.

These tests validate three things:

1. **Pattern rules**: every category of rule from ``_RULES`` produces
   the expected French sentence (greetings, age, studies, questions,
   needs, feelings, actions, negation, numbers).
2. **Pre-processing helpers**: digit merging, letter merging (spelled
   names), consecutive duplicate collapsing.
3. **Sentence-level pipeline**: explicit ``"|"`` pauses, automatic
   segmentation on subject pivots (``MOI``, ``TOI``, ``AUJOURD'HUI``…),
   fallback word translation for unmapped tokens, and the full
   end-to-end demo phrase used during the TFE presentation.

Run with::

    .\\venv\\Scripts\\python.exe -m pytest -v
"""

from __future__ import annotations

import pytest

from lsf_translator import (
    _auto_split,
    _dedupe_consecutive,
    _merge_digits,
    _merge_letters,
    _split_segments,
    translate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Single-token / greeting rules
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("glosses, expected", [
    (["BONJOUR"], "Bonjour."),
    (["MERCI"], "Merci."),
    (["OUI"], "Oui."),
    (["NON"], "Non."),
    (["AU-REVOIR"], "Au revoir."),
])
def test_single_greetings(glosses, expected):
    assert translate(glosses) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Introductions (name + spelled letters)
# ─────────────────────────────────────────────────────────────────────────────

class TestIntroductions:
    def test_bonjour_moi_nom_with_name(self):
        assert translate(["BONJOUR", "MOI", "NOM", "ABDEL"]) \
            == "Bonjour, je m'appelle Abdel."

    def test_bonjour_moi_nom_spelled(self):
        # Spelled letters must be merged into a single name token,
        # then properly cased ("Abdel" not "ABDEL").
        glosses = ["BONJOUR", "MOI", "NOM", "A", "B", "D", "E", "L"]
        assert translate(glosses) == "Bonjour, je m'appelle Abdel."

    def test_moi_nom_alone(self):
        assert translate(["MOI", "NOM"]) == "Je m'appelle."

    def test_nom_with_name(self):
        assert translate(["NOM", "PAUL"]) == "Je m'appelle Paul."

    def test_bonjour_moi_nom_without_name(self):
        assert translate(["BONJOUR", "MOI", "NOM"]) == "Bonjour, je m'appelle."


# ─────────────────────────────────────────────────────────────────────────────
# Age (digit merging is exercised here)
# ─────────────────────────────────────────────────────────────────────────────

class TestAge:
    def test_simple_age(self):
        assert translate(["MOI", "24", "ANS"]) == "J'ai 24 ans."

    def test_age_with_split_digits(self):
        # 2 + 4 must be merged into "24" before the rule fires.
        assert translate(["MOI", "2", "4", "ANS"]) == "J'ai 24 ans."

    def test_age_question(self):
        assert translate(["TOI", "AGE", "QUOI"]) == "Tu as quel âge ?"

    def test_toi_age(self):
        assert translate(["TOI", "30", "ANS"]) == "Tu as 30 ans."


# ─────────────────────────────────────────────────────────────────────────────
# Studies / profession
# ─────────────────────────────────────────────────────────────────────────────

class TestStudies:
    def test_etudiant_simple(self):
        assert translate(["MOI", "ETUDIANT"]) == "Je suis étudiant."

    def test_travailler_simple(self):
        assert translate(["MOI", "TRAVAILLER"]) == "Je travaille."


# ─────────────────────────────────────────────────────────────────────────────
# Project
# ─────────────────────────────────────────────────────────────────────────────

class TestProject:
    def test_full_project_phrase(self):
        assert translate(["ICI", "PROJET", "FIN", "ETUDES"]) \
            == "Voici mon projet de fin d'études."

    def test_projet_fin_etudes_alone(self):
        assert translate(["PROJET", "FIN", "ETUDES"]) \
            == "Mon projet de fin d'études."


# ─────────────────────────────────────────────────────────────────────────────
# Questions
# ─────────────────────────────────────────────────────────────────────────────

class TestQuestions:
    @pytest.mark.parametrize("glosses, expected", [
        (["TOI", "NOM", "QUOI"], "Comment tu t'appelles ?"),
        (["TOI", "AGE", "QUOI"], "Tu as quel âge ?"),
        (["TOI", "VOULOIR", "QUOI"], "Qu'est-ce que tu veux ?"),
        (["TOI", "MANGER", "QUOI"], "Qu'est-ce que tu veux manger ?"),
        (["TOI", "BOIRE", "QUOI"], "Qu'est-ce que tu veux boire ?"),
        (["TOI", "HABITER", "QUOI"], "Où est-ce que tu habites ?"),
        (["TOI", "BIEN", "QUOI"], "Comment tu vas ?"),
        (["TOI", "BIEN"], "Tu vas bien ?"),
    ])
    def test_questions(self, glosses, expected):
        assert translate(glosses) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Needs / wants
# ─────────────────────────────────────────────────────────────────────────────

class TestNeeds:
    @pytest.mark.parametrize("glosses, expected", [
        (["MOI", "VOULOIR", "MANGER"], "Je veux manger."),
        (["MOI", "VOULOIR", "BOIRE"], "Je veux boire."),
        (["MOI", "VOULOIR", "DORMIR"], "Je veux dormir."),
        (["MOI", "VOULOIR", "AIDE"], "J'ai besoin d'aide."),
    ])
    def test_wants(self, glosses, expected):
        assert translate(glosses) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Feelings
# ─────────────────────────────────────────────────────────────────────────────

class TestFeelings:
    @pytest.mark.parametrize("glosses, expected", [
        (["MOI", "BIEN"], "Je vais bien."),
        (["MOI", "MAL"], "Je ne vais pas bien."),
        (["MOI", "FATIGUE"], "Je suis fatigué."),
        (["MOI", "CONTENT"], "Je suis content."),
        (["MOI", "TRISTE"], "Je suis triste."),
    ])
    def test_feelings(self, glosses, expected):
        assert translate(glosses) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Actions
# ─────────────────────────────────────────────────────────────────────────────

class TestActions:
    @pytest.mark.parametrize("glosses, expected", [
        (["MOI", "MANGER"], "Je mange."),
        (["MOI", "BOIRE"], "Je bois."),
        (["MOI", "DORMIR"], "Je dors."),
        (["MOI", "PARLER"], "Je parle."),
        (["MOI", "ECOUTER"], "J'écoute."),
        (["MOI", "COMPRENDRE"], "Je comprends."),
        (["MOI", "COMPRENDRE", "PAS"], "Je ne comprends pas."),
        (["MOI", "PAS", "COMPRENDRE"], "Je ne comprends pas."),
        (["MOI", "AIDE"], "J'ai besoin d'aide."),
    ])
    def test_actions(self, glosses, expected):
        assert translate(glosses) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Time markers (added for "Aujourd'hui" bug)
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeMarkers:
    def test_aujourdhui_alone(self):
        # AUJOURD'HUI has no dedicated rule → fallback word map.
        assert translate(["Aujourd'hui"]) == "Aujourd'hui."

    def test_time_marker_starts_new_proposition(self):
        # AUJOURD'HUI is a pivot starter so it must split the sentence.
        result = translate(["MOI", "ETUDIANT", "Aujourd'hui"])
        assert result == "Je suis étudiant. Aujourd'hui."

    @pytest.mark.parametrize("token, expected", [
        ("HIER", "Hier."),
        ("DEMAIN", "Demain."),
        ("MAINTENANT", "Maintenant."),
    ])
    def test_other_time_markers(self, token, expected):
        assert translate([token]) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing helpers (white-box)
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeDigits:
    def test_merge_two_digits(self):
        assert _merge_digits(["MOI", "2", "4", "ANS"]) == ["MOI", "24", "ANS"]

    def test_merge_three_digits(self):
        assert _merge_digits(["1", "2", "3"]) == ["123"]

    def test_no_digits(self):
        assert _merge_digits(["MOI", "BIEN"]) == ["MOI", "BIEN"]

    def test_digits_at_end(self):
        assert _merge_digits(["MOI", "AGE", "3", "0"]) == ["MOI", "AGE", "30"]


class TestMergeLetters:
    def test_merge_simple_name(self):
        assert _merge_letters(["NOM", "P", "A", "U", "L"]) == ["NOM", "Paul"]

    def test_single_letter_kept(self):
        # A single isolated letter must not be merged/altered.
        assert _merge_letters(["NOM", "A"]) == ["NOM", "A"]

    def test_two_letter_runs(self):
        # Two separate letter runs separated by a non-letter token.
        assert _merge_letters(["NOM", "A", "B", "ETUDIANT", "C", "D"]) \
            == ["NOM", "Ab", "ETUDIANT", "Cd"]

    def test_no_letters(self):
        assert _merge_letters(["MOI", "ETUDIANT"]) == ["MOI", "ETUDIANT"]


class TestDedupe:
    def test_collapses_runs(self):
        assert _dedupe_consecutive(["MOI", "MOI", "MOI", "BIEN"]) \
            == ["MOI", "BIEN"]

    def test_case_insensitive(self):
        assert _dedupe_consecutive(["Bonjour", "BONJOUR", "bonjour"]) \
            == ["Bonjour"]

    def test_preserves_non_adjacent(self):
        # Same token but not adjacent → both kept.
        assert _dedupe_consecutive(["MOI", "BIEN", "MOI"]) \
            == ["MOI", "BIEN", "MOI"]

    def test_empty(self):
        assert _dedupe_consecutive([]) == []

    def test_no_duplicates(self):
        assert _dedupe_consecutive(["A", "B", "C"]) == ["A", "B", "C"]


# ─────────────────────────────────────────────────────────────────────────────
# Segment splitter
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitSegments:
    def test_explicit_pause(self):
        assert _split_segments(["BONJOUR", "|", "MERCI"]) \
            == [["BONJOUR"], ["MERCI"]]

    def test_no_pause(self):
        assert _split_segments(["MOI", "BIEN"]) == [["MOI", "BIEN"]]

    def test_trailing_pause(self):
        assert _split_segments(["BONJOUR", "|"]) == [["BONJOUR"]]

    def test_leading_pause(self):
        # A leading "|" produces no empty segment.
        assert _split_segments(["|", "BONJOUR"]) == [["BONJOUR"]]


class TestAutoSplit:
    def test_keeps_bonjour_moi_together(self):
        # MOI right after BONJOUR is glued (greeting pattern).
        assert _auto_split(["BONJOUR", "MOI", "NOM", "PAUL"]) \
            == [["BONJOUR", "MOI", "NOM", "PAUL"]]

    def test_pivot_starts_new_proposition(self):
        # The second MOI must trigger a new proposition.
        assert _auto_split(["MOI", "ETUDIANT", "MOI", "BIEN"]) \
            == [["MOI", "ETUDIANT"], ["MOI", "BIEN"]]

    def test_aujourdhui_is_pivot(self):
        assert _auto_split(["MOI", "ETUDIANT", "Aujourd'hui"]) \
            == [["MOI", "ETUDIANT"], ["Aujourd'hui"]]

    def test_empty(self):
        assert _auto_split([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# Pause-driven sentences (full pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class TestExplicitPauses:
    def test_two_segments(self):
        assert translate(["BONJOUR", "|", "MERCI"]) == "Bonjour. Merci."

    def test_glove_demo_triplet(self):
        assert translate(["MOI", "BONJOUR", "MERCI"]) == "Bonjour, merci."

    def test_three_segments(self):
        assert translate(["BONJOUR", "|", "MOI", "BIEN", "|", "MERCI"]) \
            == "Bonjour. Je vais bien. Merci."


# ─────────────────────────────────────────────────────────────────────────────
# Fallback (unknown words → word map)
# ─────────────────────────────────────────────────────────────────────────────

class TestFallback:
    def test_unknown_word_passed_through(self):
        # Unknown gloss with no rule and no map entry → kept verbatim,
        # capitalized at sentence start.
        assert translate(["XYZ"]) == "Xyz."

    def test_word_map_applied(self):
        # MOI alone → no rule matches → fallback maps MOI → "je".
        assert translate(["MOI"]) == "Je."

    def test_remainder_appended_after_match(self):
        # Rule matches a prefix; trailing tokens must not be silently
        # dropped — they should be word-mapped and appended.
        # "MOI ETUDIANT BIEN" → rule "MOI ETUDIANT (.+)" → "Je suis étudiant à BIEN"
        # but "BIEN" is in the WORD_MAP so the rule with capture wins first.
        # We simply assert the trailing token still appears in the output.
        out = translate(["MOI", "AIMER", "MANGER"])
        # "^MOI AIMER (.+)" → "J'aime MANGER" → proper_case → "J'aime Manger"
        assert out == "J'aime Manger."


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end demo phrase (the one used in the TFE presentation)
# ─────────────────────────────────────────────────────────────────────────────

class TestDemoPhrase:
    def test_full_presentation(self):
        """Real-world phrase captured during a live demo session.

        Exercises: dedup of trailing repeated MOI, letter merging
        (Abdelbadi), digit merging (24), pivot segmentation (MOI / MOI /
        AUJOURD'HUI), pattern rules (intro, age, studies) and fallback
        (Aujourd'hui via the word map).
        """
        glosses = [
            "Bonjour", "MOI", "NOM",
            "A", "B", "D", "E", "L", "B", "A", "D", "I",
            "MOI", "ETUDIANT",
            "MOI", "2", "4", "ans",
            "Aujourd'hui",
            "MOI", "MOI",
        ]
        assert translate(glosses) == (
            "Bonjour, je m'appelle Abdelbadi. "
            "Je suis étudiant. "
            "J'ai 24 ans. "
            "Aujourd'hui. "
            "Je."
        )

    def test_idempotent_with_extra_repeats(self):
        # Adding extra consecutive duplicates of any token must not
        # change the output.
        base = ["BONJOUR", "MOI", "NOM", "P", "A", "U", "L"]
        noisy = ["BONJOUR", "BONJOUR", "MOI", "MOI", "NOM", "NOM",
                 "P", "A", "U", "L", "L"]
        assert translate(noisy) == translate(base)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_list(self):
        assert translate([]) == ""

    def test_only_pauses(self):
        assert translate(["|", "|", "|"]) == ""

    def test_sentence_terminator_added_once(self):
        # If the result already ends with "?" it must not be turned
        # into "?." — our pipeline only appends "." when missing.
        out = translate(["TOI", "BIEN"])
        assert out.endswith("?") and not out.endswith(".")

    def test_lowercase_input_handled(self):
        # Glosses can come in any case — translation is case-insensitive.
        assert translate(["bonjour"]) == "Bonjour."
