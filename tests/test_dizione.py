"""Tests of dizione.

The first classes need nothing but the source. The last two need the built
data (dictionary, model, homograph weights): set DIZIONE_DATA to the built
directory or put it in ``src/dizione/built``; without it they are skipped.
"""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

import dizione as phonemizer
from dizione import align, convert, homographs, morphology, normalize, sentence, tables
from dizione.engine import Phonemizer, default_directory
from dizione.sentence import Unit
from dizione.symbols import PHONE_SET, SP, SYMBOL_TO_ID, SYMBOLS, VOWEL_SET, tokenize

ROOT = Path(__file__).resolve().parents[1]
BUILT = default_directory()
HAVE_DATA = (BUILT / "lexicon.tsv").exists() and (BUILT / "g2p.npz").exists() and (BUILT / "homographs.json").exists()
P = convert.from_text


def marked(result):
    """Token sequence as text: stress marks on the phones, <sp> between words."""
    return " ".join({1: "ˈ", 2: "ˌ"}.get(level, "") + symbol for symbol, level in zip(result.symbols[1:-1], result.stress[1:-1]))


def said(tokens):
    out = []
    for token in tokens:
        if isinstance(token, normalize.Punct):
            out.append(token.mark)
            continue
        shown = token.text
        if token.spell:
            shown = "[" + " ".join(shown.upper()) + "]"
        if token.phones is not None:
            shown = "{" + " ".join(phone for phone, _ in token.phones) + "}"
        out.append(("+" if token.attach else "") + shown)
    return " ".join(out)


class ConversionTests(unittest.TestCase):
    """CMUdict notation -> phones of the inventory, one word."""

    CASES = (
        ("water", "W AO1 T ER0", "w|a|t|er", "w ˈɔː ɾ ɚ"),                     # flap
        ("party", "P AA1 R T IY0", "p|a|r|t|y", "p ˈɑːɹ ɾ i"),                  # flap after an r-coloured vowel
        ("potato", "P AH0 T EY1 T OW2", "p|o|t|a|t|o", "p ə t ˈeɪ ɾ oʊ"),       # flap before a final o
        ("wanted", "W AO1 N T IH0 D", "w|a|n|t|e|d", "w ˈɔ n t ᵻ d"),           # no flap after n; -ed
        ("important", "IH2 M P AO1 R T AH0 N T", "i|m|p|o|r|t|a|n|t", "ɪ m p ˈoːɹ t ə n t"),
        ("button", "B AH1 T AH0 N", "b|u|tt|o|n", "b ˈʌ ʔ n̩"),                  # glottal stop + syllabic n
        ("little", "L IH1 T AH0 L", "l|i|tt||le", "l ˈɪ ɾ əl"),                  # syllabic l
        ("finally", "F AY1 N AH0 L IY0", "f|i|n|a|ll|y", "f ˈaɪ n əl i"),
        ("family", "F AE1 M AH0 L IY0", "f|a|m|i|l|y", "f ˈæ m ɪ l i"),         # reduced vowel written i
        ("nation", "N EY1 SH AH0 N", "n|a|t|io|n", "n ˈeɪ ʃ ə n"),              # ... but not "io"
        ("possible", "P AA1 S AH0 B AH0 L", "p|o|ss|i|b||le", "p ˈɑː s ᵻ b əl"),
        ("about", "AH0 B AW1 T", "a|b|ou|t", "ɐ b ˈaʊ t"),                      # initial a
        ("expect", "IH0 K S P EH1 K T", "e|x||p|e|c|t", "ɛ k s p ˈɛ k t"),
        ("before", "B IH0 F AO1 R", "b|e|f|o|re", "b ᵻ f ˈoːɹ"),                # weak prefix; FORCE
        ("eleven", "IH0 L EH1 V AH0 N", "e|l|e|v|e|n", "ɪ l ˈɛ v ə n"),
        ("houses", "HH AW1 Z AH0 Z", "h|ou|s|e|s", "h ˈaʊ z ᵻ z"),              # -es after a sibilant
        ("more", "M AO1 R", "m|o|re", "m ˈoːɹ"),
        ("for", "F AO1 R", "f|o|r", "f ˈɔːɹ"),                                  # NORTH
        ("story", "S T AO1 R IY0", "s|t|o|r|y", "s t ˈoː ɹ i"),                 # r-coloured vowel split before a vowel
        ("hearing", "HH IY1 R IH0 NG", "h|ea|r|i|ng", "h ˈɪ ɹ ɪ ŋ"),
        ("zero", "Z IH1 R OW0", "z|e|r|o", "z ˈiə ɹ oʊ"),
        ("during", "D UH1 R IH0 NG", "d|u|r|i|ng", "d ˈʊɹ ɹ ɪ ŋ"),
        ("marry", "M EH1 R IY0", "m|a|rr|y", "m ˈæ ɹ i"),
        ("very", "V EH1 R IY0", "v|e|r|y", "v ˈɛ ɹ i"),
        ("fire", "F AY1 ER0", "f|i|re", "f ˈaɪɚ"),
        ("higher", "HH AY1 ER0", "h|igh|er", "h ˈaɪ ɚ"),
        ("around", "ER0 AW1 N D", "ar|ou|n|d", "ɚ ɹ ˈaʊ n d"),                  # linking r inside a word
        ("quiet", "K W AY1 AH0 T", "q|u|i|e|t", "k w ˈaɪə t"),
        ("idea", "AY0 D IY1 AH0", "i|d|e|a", "aɪ d ˈiə"),
        ("opinion", "AH0 P IH1 N Y AH0 N", "o|p|i|n|i|o|n", "ə p ˈɪ n iə n"),
        ("popular", "P AA1 P Y AH0 L ER0", "p|o|p||u|l|ar", "p ˈɑː p j ʊ l ɚ"),
        ("usual", "Y UW1 ZH AH0 W AH0 L", "|u|s|u||a|l", "j ˈuː ʒ uː əl"),
        ("long", "L AO1 NG", "l|o|ng", "l ˈɔ ŋ"),                               # CLOTH
        ("law", "L AO1", "l|aw", "l ˈɔː"),                                      # THOUGHT
        ("want", "W AA1 N T", "w|a|n|t", "w ˈɔ n t"),
        ("income", "IH1 N K AH2 M", "i|n|c|o|me", "ˈɪ ŋ k ʌ m"),                # n before k; no stress next to the primary
        ("weekend", "W IY1 K EH2 N D", "w|ee|k|e|n|d", "w ˈiː k ɛ n d"),
        ("everything", "EH1 V R IY0 TH IH2 NG", "e|ve|r|y|th|i|ng", "ˈɛ v ɹ ɪ θ ˌɪ ŋ"),
        ("happy", "HH AE1 P IY0", "h|a|pp|y", "h ˈæ p i"),
        ("stories", "S T AO1 R IY0 Z", "s|t|o|r|ie|s", "s t ˈoː ɹ i z"),
    )

    def test_rules_on_aligned_words(self):
        for word, arpabet, runs, expected in self.CASES:
            with self.subTest(word=word):
                phones = convert.convert(word, arpabet.split(), runs.split("|"))
                self.assertEqual("".join(runs.split("|")), word)
                self.assertEqual(convert.to_text(phones), expected)
                self.assertTrue(all(phone in PHONE_SET for phone, _ in phones))
                self.assertTrue(all(phone in VOWEL_SET for phone, stress in phones if stress))

    def test_without_spelling_the_plain_rules_still_apply(self):
        self.assertEqual(convert.convert("", "HH AH0 L OW1".split()), P("h ə l ˈoʊ"))
        self.assertEqual(convert.to_text(convert.convert("", "W AO1 T ER0".split())), "w ˈɔː ɾ ɚ")
        with self.assertRaises(ValueError):
            convert.convert("", ["XX1"])

    def test_phone_text_round_trip_and_validation(self):
        phones = P("h ə l ˈoʊ w ˌɜː l d")
        self.assertEqual(convert.from_text(convert.to_text(phones)), phones)
        self.assertTrue(convert.is_arpabet("HH AH0 L OW1".split()) and not convert.is_arpabet("h ə l oʊ".split()))
        for bad in ("h e l o", "ˈh ə"):                    # a symbol outside the inventory; stress on a consonant
            with self.assertRaises(ValueError):
                convert.from_text(bad)

    def test_one_primary_stress_and_no_secondary_next_to_it(self):
        self.assertEqual(convert.to_text(convert.settle_stress(P("d ˈaʊ n t ˈaʊ n"))), "d aʊ n t ˈaʊ n")
        self.assertEqual(convert.to_text(convert.settle_stress(P("ˌæ f t ɚ n ˈuː n"))), "ˌæ f t ɚ n ˈuː n")


class AlignmentTests(unittest.TestCase):
    def test_letters_are_assigned_to_phonemes_in_order(self):
        entries = [("that", "DH AE1 T"), ("this", "DH IH1 S"), ("thin", "TH IH1 N"), ("bath", "B AE1 TH"), ("cat", "K AE1 T"),
                   ("cab", "K AE1 B"), ("bat", "B AE1 T"), ("tab", "T AE1 B"), ("sit", "S IH1 T"), ("bit", "B IH1 T"),
                   ("tin", "T IH1 N"), ("nab", "N AE1 B"), ("box", "B AA1 K S"), ("fox", "F AA1 K S"), ("cot", "K AA1 T"),
                   ("knit", "N IH1 T"), ("knot", "N AA1 T"), ("knob", "N AA1 B")]
        table = align.train([(word, phones.split()) for word, phones in entries], iterations=6)
        aligner = align.Aligner(table)
        for word, phones in entries:
            runs = aligner.align(word, phones.split())
            self.assertEqual(len(runs), len(phones.split()))
            self.assertEqual("".join(runs), word)
        self.assertEqual(aligner.align("that", ["DH", "AE1", "T"]), ["th", "a", "t"])
        self.assertEqual(aligner.align("knit", ["N", "IH1", "T"]), ["kn", "i", "t"])
        self.assertEqual(aligner.align("box", ["B", "AA1", "K", "S"])[:2], ["b", "o"])
        self.assertIsNone(aligner.align("abbreviation", ["EY1"]))          # too many letters for the phonemes


class NormalizerTests(unittest.TestCase):
    """Rules that need no dictionary (every spelling counts as a word)."""

    def test_numbers_as_words(self):
        self.assertEqual(normalize.cardinal(0), "zero")
        self.assertEqual(normalize.cardinal(21), "twenty one")
        self.assertEqual(normalize.cardinal(1234567), "one million two hundred thirty four thousand five hundred sixty seven")
        self.assertEqual([normalize.ordinal(n) for n in (1, 2, 3, 12, 20, 21, 101)],
                         ["first", "second", "third", "twelfth", "twentieth", "twenty first", "one hundred first"])
        self.assertEqual([normalize.year(n) for n in (1984, 1905, 1900, 2000, 2005, 2026)],
                         ["nineteen eighty four", "nineteen oh five", "nineteen hundred", "two thousand", "two thousand five",
                          "twenty twenty six"])
        self.assertEqual(normalize.number_words("3.14"), "three point one four")
        self.assertEqual(normalize.number_words("007"), "zero zero seven")

    def test_money_time_date_and_symbols(self):
        for written, spoken in (
                ("It costs $3.14.", "it costs three dollars and fourteen cents ."),
                ("At 3:05 PM.", "at three oh five [P M] ."),
                ("On 2026-11-14.", "on november fourteenth twenty twenty six ."),
                ("Call 1-800-555-0147.", "call one , eight hundred , five five five , oh one four seven ."),
                ("45% & more", "forty five percent and more")):
            self.assertEqual(said(normalize.tokens(written)), spoken)

    def test_versions_codes_fractions_and_lengths_of_time(self):
        for written, spoken in (
                ("Update to v2.4.1.", "update to [V] two point four point one ."),
                ("It took 2:03:45.", "it took two hours , three minutes and forty five seconds ."),
                ("Plate KT-738-RP, bug #2093.", "plate [K T] , seven three eight , [R P] , bug number two zero nine three ."),
                ("Card 4502 1187 3390 2.", "card four five zero two , one one eight seven , three three nine zero , two ."),
                ("Add 1½ cups.", "add one and a half cups ."),
                ("x ≥ 5", "[X] greater than or equal to five"),
                ("On 14/03 at 20:00.", "on march fourteenth at twenty hundred .")):
            self.assertEqual(said(normalize.tokens(written)), spoken)
        self.assertEqual(normalize.pairs(4070), "forty seventy")
        self.assertEqual(normalize.duration(1, 0, 5), "one hour and five seconds")

    def test_continental_figures(self):
        """"1.299,50": a point between the thousands, a comma before the decimals."""
        euros = "one thousand two hundred ninety nine euros and fifty cents"
        for written, spoken in (
                ("It costs 1.299,50 today.", "it costs one thousand two hundred ninety nine point five zero today ."),
                ("€1.299,50, 1.299,50 € and 12,50 EUR.", f"{euros} , {euros} and twelve euros and fifty cents ."),
                ("Up 3,5% to 1.234.567,89.", "up three point five percent to one million two hundred thirty four thousand "
                                             "five hundred sixty seven point eight nine ."),
                ("A 1,5 kg bag for 5-10 €, not -3 €.", "a one point five kilograms bag for five to ten euros , not minus three euros ."),
                # a comma that parts thousands or the items of a list stays one, and so does a point before decimals
                ("$1,250.50, 1,234 and 3.14.", "one thousand two hundred fifty dollars and fifty cents , "
                                               "one thousand two hundred thirty four and three point one four ."),
                ("Rooms 1,2 and 3; 2.500 people; €12,500.", "rooms one , two and three ; two point five zero zero people ; "
                                                            "twelve thousand five hundred euros ."),
                ("Pay 3,50 or 4,20 at 192.168.100.200,80.", "pay three , fifty or four , twenty at one nine two point one six eight "
                                                            "point one zero zero point two zero zero , eighty .")):
            self.assertEqual(said(normalize.tokens(written)), spoken)

    def test_emoticons_and_emoji_are_neither_words_nor_pauses(self):
        for written in ("ok :))", "ok :-) <3", "ok 🙂👍", "ok :D ;) :P"):
            tokens = normalize.tokens(written)
            self.assertEqual(said(tokens), "okay")
            self.assertFalse(any(isinstance(token, normalize.Punct) for token in tokens))

    def test_inline_pronunciation(self):
        tokens = normalize.tokens("Say {HH AH0 L OW1} to Samir{s ə m ˈɪɹ}.")
        given = [token.phones for token in tokens if isinstance(token, normalize.Word) and token.phones]
        self.assertEqual(given, [P("h ə l ˈoʊ"), P("s ə m ˈɪɹ")])
        with self.assertRaises(ValueError):
            normalize.tokens("Say {x y z}.")

    def test_stretched_words(self):
        self.assertEqual(normalize.Normalizer._unstretched("soooo")[0], "so")
        self.assertEqual(normalize.Normalizer._unstretched("goooood")[0], "good")
        self.assertEqual(normalize.Normalizer._unstretched("tooo")[0], "too")
        self.assertEqual(normalize.Normalizer._unstretched("alll")[0], "all")
        self.assertEqual(normalize.Normalizer._unstretched("mmmm")[0], "mm")       # the hum, not the letter M
        self.assertEqual(normalize.Normalizer._unstretched("sirr"), [])

    def test_a_hiss_and_a_snore_only_where_they_stand_apart(self):
        tokens = normalize.tokens                           # every spelling counts as a word here
        self.assertEqual(said(tokens("Sh! Ss! Tsk!")), "sh ! {s} ! tsk !")
        self.assertEqual(said(tokens("Sh ss tsk, Sss, Zz. Zzzz")), "sh {s} tsk , {s} , {z} . {z}")
        self.assertEqual(said(tokens('"Ss!" she hissed.')), '" {s} ! " she hissed .')
        # Beside a word, joined to a mark or in capitals, the letters are read as they always were.
        for written, spoken in (("The format is mm:ss.", "the format is mm : ss ."), ("the ss uniforms", "the ss uniforms"),
                                ("Write ss here.", "write ss here ."), ("p*ss", "[P] ss"), ("return ss.str()", "return ss . str ( )"),
                                ('He wrote "ss" there.', 'he wrote " ss " there .'),
                                ("SS and ZZ", "ss and zz"), ("SSDs", "ssds"), ("catching some zzz's", "catching some zzz's")):
            self.assertEqual(said(tokens(written)), spoken)
        self.assertEqual(said(tokens("THE SSS SOUND IS LOUD")), "the {s} sound is loud")       # a text typed in capitals

    def test_a_currency_sign_after_the_amount(self):
        for written, spoken in (
                ("I paid 100$ and 20 $, then 5 $, 10 $ and 3 £.",
                 "i paid one hundred dollars and twenty dollars , then five dollars , ten dollars and three pounds ."),
                ("From $5 to 30$, or 1.25$ to 1.40$.",
                 "from five dollars to thirty dollars , or one dollar and twenty five cents to one dollar and forty cents ."),
                # thousands parted by spaces
                ("A budget of 750 000 $, 51 243 € each, 1 299,50 € more.",
                 "a budget of seven hundred fifty thousand dollars , fifty one thousand two hundred forty three euros each , "
                 "one thousand two hundred ninety nine euros and fifty cents more ."),
                # the sign that closes a formula is not a currency: these are read as before
                ("Let $x = 2$ and $E=mc^2$, so $3 + 5 = 8$.",
                 "let [X] equals two and [E] equals mc two , so three dollars plus five equals eight ."),
                ("The variance $\\sigma^2$ with $n_0 = 1$.", "the variance sigma two with [N] zero equals one ."),
                ("The area is $(a + b)^2$ and $c^2$.", "the area is ( a plus [B] ) two and [C] two ."),
                ("So x^2$ and 5^2 $ are not prices.", "so [X] two and five to the power of two are not prices ."),
                ("The $F_1$-Score is 0.9.", "the [F] one +score is zero point nine ."),
                ("An area of 1,0 m² or 1,5 m².", "an area of one point zero square meter or one point five square meters .")):
            self.assertEqual(said(normalize.tokens(written)), spoken)


class SentenceTests(unittest.TestCase):
    def units(self, *spec):
        return [Unit(word, P(form), kind, P(final) if final else None) for word, kind, form, final in spec]

    def test_function_words_lose_or_keep_stress_by_position(self):
        units = self.units(("i", "uf", "aɪ", None), ("want", "c", "w ˈɔ n t", None), ("to", "uf", "t ə", "t uː"),
                           ("go", "s", "ɡ ˌoʊ", None), ("there", "uf", "ð ɛɹ", None))
        sentence.stress_clause(units)
        self.assertEqual([convert.to_text(u.phones) for u in units], ["aɪ", "w ˈɔ n t", "t ə", "ɡ ˌoʊ", "ð ˈɛɹ"])
        units = self.units(("where", "s", "w ˌɛɹ", None), ("to", "uf", "t ə", "t uː"))
        sentence.stress_clause(units)                      # nothing stressed follows "where"; "to" closes the clause
        self.assertEqual([convert.to_text(u.phones) for u in units], ["w ˈɛɹ", "t uː"])
        units = self.units(("what", "si", "w ˌʌ t", None), ("is", "u", "ɪ z", None), ("it", "u", "ɪ t", None),
                           ("what", "si", "w ˌʌ t", None), ("for", "u", "f ɔːɹ", None), ("now", "c", "n ˈaʊ", None))
        sentence.stress_clause(units)
        self.assertEqual([convert.to_text(u.phones) for u in units], ["w ˌʌ t", "ɪ z", "ɪ t", "w ʌ t", "f ɔːɹ", "n ˈaʊ"])
        units = self.units(("one", "pf", "w ˈʌ n", "w ˌʌ n"), ("more", "c", "m ˈoːɹ", None), ("one", "pf", "w ˈʌ n", "w ˌʌ n"))
        sentence.stress_clause(units)
        self.assertEqual([convert.to_text(u.phones) for u in units], ["w ˈʌ n", "m ˈoːɹ", "w ˌʌ n"])

    def test_a_clause_is_never_left_without_stress(self):
        units = self.units(("it", "u", "ɪ t", None), ("is", "u", "ɪ z", None))
        sentence.stress_clause(units)
        self.assertEqual([convert.to_text(u.phones) for u in units], ["ɪ t", "ˈɪ z"])

    def test_words_adjust_to_the_next_word(self):
        units = self.units(("the", "u", "ð ə", None), ("apple", "c", "ˈæ p əl", None), ("for", "u", "f ɔːɹ", None),
                           ("it", "u", "ɪ t", None), ("is", "u", "ɪ z", None), ("after", "c", "ˈæ f t ɚ", None),
                           ("all", "c", "ˈɔː l", None), ("to", "u", "t ə", None), ("eat", "c", "ˈiː t", None))
        sentence.join_clause(units)
        self.assertEqual([convert.to_text(u.phones) for u in units],
                         ["ð ɪ", "ˈæ p əl", "f ɔː ɹ", "ɪ ɾ", "ɪ z", "ˈæ f t ɚ ɹ", "ˈɔː l", "t ʊ", "ˈiː t"])
        units = self.units(("her", "u", "h ɜː", None), ("and", "uf", "æ n d", None), ("but", "s", "b ˌʌ t", None),
                           ("i", "uf", "aɪ", None))
        sentence.join_clause(units)                        # nothing links into "and"; only "it" and "at" flap
        self.assertEqual([convert.to_text(u.phones) for u in units], ["h ɜː", "æ n d", "b ˌʌ t", "aɪ"])

    def test_attached_units_form_one_spoken_word(self):
        units = [Unit("well", P("w ˈɛ l")), Unit("known", P("n ˈoʊ n"), attach=True), Unit("thing", P("θ ˈɪ ŋ"))]
        self.assertEqual([convert.to_text(group) for group in sentence.groups(units)], ["w ˈɛ l n ˈoʊ n", "θ ˈɪ ŋ"])


class MorphologyTests(unittest.TestCase):
    WORDS = {"wait": "w ˈeɪ t", "star": "s t ˈɑːɹ", "happy": "h ˈæ p i", "dog": "d ˈɔ ɡ", "cat": "k ˈæ t", "bus": "b ˈʌ s",
             "sand": "s ˈæ n d", "piper": "p ˈaɪ p ɚ", "herb": "ˈɜː b", "copa": "k ˈoʊ p ə", "coup": "k ˈuː", "let": "l ˈɛ t",
             "walk": "w ˈɔː k", "love": "l ˈʌ v", "final": "f ˈaɪ n əl", "web": "w ˈɛ b", "cam": "k ˈæ m", "wander": "w ˈɔ n d ɚ",
             "story": "s t ˈoː ɹ i", "stop": "s t ˈɑː p", "hope": "h ˈoʊ p", "hop": "h ˈɑː p", "kind": "k ˈaɪ n d"}

    def derive(self, word):
        found = morphology.derive(word, lambda spelling: P(self.WORDS[spelling]) if spelling in self.WORDS else None)
        return convert.to_text(found) if found else None

    def test_regular_endings(self):
        for word, expected in (("waited", "w ˈeɪ ɾ ᵻ d"), ("waiting", "w ˈeɪ ɾ ɪ ŋ"), ("starring", "s t ˈɑː ɹ ɪ ŋ"),
                               ("wandering", "w ˈɔ n d ɚ ɹ ɪ ŋ"), ("dogs", "d ˈɔ ɡ z"), ("cats", "k ˈæ t s"),
                               ("buses", "b ˈʌ s ᵻ z"), ("cat's", "k ˈæ t s"), ("walked", "w ˈɔː k t"), ("loved", "l ˈʌ v d"),
                               ("stopped", "s t ˈɑː p t"), ("hoped", "h ˈoʊ p t"), ("hopped", "h ˈɑː p t"),
                               ("stories", "s t ˈoː ɹ i z"), ("happily", "h ˈæ p ɪ l i"), ("finally", "f ˈaɪ n əl i"),
                               ("happiness", "h ˈæ p ɪ n ə s"), ("unkind", "ʌ ŋ k ˈaɪ n d"), ("herbage", "ˈɜː b ɪ dʒ"),
                               ("sandpiper", "s ˈæ n d p ˌaɪ p ɚ")):
            with self.subTest(word=word):
                self.assertEqual(self.derive(word), expected)
        for word in ("zzzz", "webcam", "couplet", "copay"):    # short halves and stray endings are not trusted
            self.assertIsNone(self.derive(word), word)


class TableTests(unittest.TestCase):
    def test_hand_written_tables_are_well_formed(self):
        functions = tables.function_words()
        self.assertGreater(len(functions), 120)
        for word, (kind, form, final) in functions.items():
            self.assertIn(kind, tables.STRESS_CLASSES, word)
            self.assertTrue(any(phone in VOWEL_SET for phone, _ in form), word)
            if kind in ("u", "uf", "ur"):
                self.assertFalse(any(stress for _, stress in form), word)       # written unstressed
            if kind in ("s", "ss", "si"):
                self.assertTrue(any(stress == 2 for _, stress in form), word)   # written with the secondary mark
        for words, (kind, forms, final) in tables.phrases().items():
            self.assertGreaterEqual(len(words), 2)
            self.assertIn(len(forms), (1, len(words)))
        letters = tables.mapping("letters.txt")
        self.assertEqual(sorted(letters), list("abcdefghijklmnopqrstuvwxyz"))
        for name in letters.values():
            self.assertEqual(sum(stress == 1 for _, stress in P(name)), 1)
        for word, (kind, value) in tables.pronunciations("exceptions.txt").items():
            self.assertTrue(kind in ("arpabet", "phones") and value, word)
        self.assertGreater(len(homographs.read_rules()), 5)
        normalize.Normalizer()                              # abbreviations, units and chat tables load

    def test_user_overrides_accept_both_notations(self):
        table = tables.pronunciations({"Samir": "S AH0 M IH1 R", "kubernetes": "k ˌuː b ɚ n ˈɛ ɾ iː z"})
        self.assertEqual(table["samir"], ("arpabet", ["S", "AH0", "M", "IH1", "R"]))
        self.assertEqual(table["kubernetes"][0], "phones")
        with self.assertRaises(ValueError):
            tables.pronunciations({"bad": "x y z"})

    def test_a_name_of_several_words_has_one_pronunciation_per_word(self):
        source = {"Tung Tung Sahur": "T UW1 NG | T UW2 NG | S AA0 HH UH1 R", "samir": "S AH0 M IH1 R"}
        table = tables.names(source)
        self.assertEqual(list(table), [("tung", "tung", "sahur")])
        self.assertEqual([kind for kind, _ in table[("tung", "tung", "sahur")]], ["arpabet"] * 3)
        self.assertEqual(list(tables.pronunciations(source)), ["samir"])
        with self.assertRaises(ValueError):
            tables.names({"tung sahur": "T UW1 NG"})
        self.assertIn(("tung", "tung", "tung", "sahur"), tables.names("exceptions.txt"))

    def test_a_homograph_rule_can_ask_for_a_hyphen_before_the_word(self):
        self.assertTrue(any(sense == "past" and name == "joined" and "well" in words
                            for sense, conditions in homographs.read_rules()["read"] for name, words in conditions))
        well, tokens = frozenset(("well", "widely")), ["a", "well", "read", "girl"]
        self.assertTrue(homographs._holds("joined", well, tokens, 2, True))
        self.assertFalse(homographs._holds("joined", well, tokens, 2))             # "as well read it": no hyphen
        self.assertFalse(homographs._holds("joined", well, ["a", "must", "read"], 2, True))
        self.assertFalse(homographs._holds("joined", well, ["read", "it"], 0, True))
        with self.assertRaises(ValueError):
            homographs.read_rules("read  past  hyphen: well")


class InventoryTests(unittest.TestCase):
    def test_the_inventory_is_fixed(self):
        self.assertEqual((len(SYMBOLS), len(PHONE_SET), len(set(SYMBOLS))), (76, 59, 76))
        self.assertEqual(SYMBOLS[:5], ("<pad>", "<bos>", "<eos>", "<sp>", "<unk>"))
        self.assertEqual(phonemizer.VERSION, 4)


class ModelArithmeticTests(unittest.TestCase):
    def test_numpy_forward_pass_equals_the_trained_network(self):
        """The run-time model is NumPy only; it must compute what PyTorch trained."""
        import torch
        spec = importlib.util.spec_from_file_location("train_g2p", ROOT / "scripts" / "train_g2p.py")
        trainer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(trainer)
        from dizione.g2p import Model
        torch.manual_seed(3)
        symbols = ["AH0", "AH1", "B", "K", "L", "OW1", "S", "T"]
        for settings in ({"width": 12, "hidden": 10, "convs": 1, "kernel": 3, "layers": 2, "slots": 2},
                         {"width": 8, "hidden": 6, "convs": 0, "kernel": 3, "layers": 1, "slots": 3}):
            network = trainer.G2P(len(symbols), dropout=0.0, **settings).eval()
            with tempfile.TemporaryDirectory() as directory:
                trainer.export(network, symbols, Path(directory) / "g2p", {})
                model = Model(directory)
                self.assertEqual(model.parameters, sum(p.numel() for p in network.parameters()))
                for word in ("a", "box", "don't", "tablecloths"):
                    letters = torch.tensor([trainer.encode_word(word)])
                    with torch.no_grad():
                        expected, _ = network(letters, torch.tensor([len(word)]))
                    np.testing.assert_allclose(model.log_probabilities(word), expected[0].numpy(), atol=2e-5)
                    self.assertEqual(model.predict(word), [symbols[k - 1] for k in trainer.greedy(expected, torch.tensor([len(word) * settings["slots"]]))[0]])
                self.assertEqual(model.spellable("Café's"), "cafe's")


@unittest.skipUnless(HAVE_DATA, "needs the built data (DIZIONE_DATA)")
class TrickySentenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phonemizer = Phonemizer(BUILT)
        cls.blocks, block = [], None
        for line in (ROOT / "tests" / "data" / "tricky_sentences.txt").read_text(encoding="utf-8").splitlines():
            if line.startswith("TEXT: "):
                block = {"text": line[6:], "said": None, "phones": []}
                cls.blocks.append(block)
            elif line.startswith("SAID: "):
                block["said"] = line[6:]
            elif line.startswith("PHONES: "):
                block["phones"].append(line[8:])

    def test_every_block(self):
        self.assertGreater(len(self.blocks), 100)
        for block in self.blocks:
            with self.subTest(text=block["text"]):
                if block["said"] is not None:
                    self.assertEqual(said(self.phonemizer.normalizer.tokens(block["text"])), block["said"])
                result = tokenize(self.phonemizer.pieces(block["text"]))
                self.assertEqual(result.unknown, ())
                self.assertTrue(all(symbol in SYMBOLS for symbol in result.symbols))
                self.assertEqual(len(result.symbols), len(result.stress))
                for stretch in block["phones"]:
                    self.assertIn(" " + stretch + " ", " " + marked(result) + " ")


@unittest.skipUnless(HAVE_DATA, "needs the built data (DIZIONE_DATA)")
class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phonemizer = Phonemizer(BUILT)

    def tearDown(self):
        phonemizer.reset()

    def speak(self, sentence_text, engine=None):
        return marked(tokenize((engine or self.phonemizer).pieces(sentence_text)))

    def test_a_greeting(self):
        self.assertEqual(self.speak("Hello Samir"), "h ə l ˈoʊ <sp> s ə m ˈɪɹ")

    def test_the_public_interface(self):
        import os
        os.environ[phonemizer_variable()] = str(BUILT)
        text = phonemizer
        result = text.phonemize("Hello Samir, it costs $3.50.")
        self.assertEqual(marked(result), "h ə l ˈoʊ <sp> s ə m ˈɪɹ , ɪ t <sp> k ˈɔ s t s <sp> θ ɹ ˈiː <sp> d ˈɑː l ɚ z <sp> "
                                         "æ n d <sp> f ˈɪ f t i <sp> s ˈɛ n t s .")
        self.assertEqual(result.ids, tuple(SYMBOL_TO_ID[s] for s in result.symbols))
        self.assertEqual(result.unknown, ())
        self.assertEqual(text.phonemize("You did.").ids, text.phonemize(" You  did. ").ids)
        self.assertNotEqual(text.phonemize("You did.").ids, text.phonemize("You did?").ids)
        with self.assertRaises(ValueError):
            text.phonemize(":)")                           # nothing to pronounce
        self.assertEqual(str(text.phonemize("Hello")), "<bos> h ə l ˈoʊ <eos>")

    def test_user_overrides_come_first(self):
        engine = Phonemizer(BUILT, overrides={"samir": "S AE1 M ER0", "read": "ɹ ˈɛ d", "the": "ð ˈiː"})
        self.assertEqual(self.speak("Samir will read the book", engine),
                         "s ˈæ m ɚ <sp> w ɪ l <sp> ɹ ˈɛ d <sp> ð ˈiː <sp> b ˈʊ k")
        self.assertEqual(self.speak("Samir will read the book"),
                         "s ə m ˈɪɹ <sp> w ɪ l <sp> ɹ ˈiː d <sp> ð ə <sp> b ˈʊ k")

    def test_a_taught_name_keeps_its_rhythm(self):
        self.assertEqual(self.speak("Tung Tung Tung Sahur"), "t ˈuː ŋ <sp> t ˌuː ŋ <sp> t ˌuː ŋ <sp> s ɑː h ˈʊɹ")
        self.assertEqual(self.speak("Tung, sahur!"), "t ˈuː ŋ , s ɑː h ˈʊɹ !")
        engine = Phonemizer(BUILT, overrides={"skibidi toilet": "S K IH2 B IH0 D IY0 | T OY1 L AH0 T", "sahur": "S AA1 HH ER0"})
        self.assertIn("s k ˌɪ b ɪ d i <sp> t ˈɔɪ l ə t", self.speak("A skibidi toilet", engine))
        self.assertIn("s ˈɑː h ɚ", self.speak("Tung tung tung sahur", engine))      # the user's own word wins over a built-in name

    def test_unknown_words_are_pronounced_by_the_model_not_spelled(self):
        for word in ("flurbish", "zentrix", "gwynplaine", "barkilphedro"):
            phones, source = self.phonemizer.word(word)
            self.assertEqual(source, "model")
            self.assertGreaterEqual(sum(phone in VOWEL_SET for phone, _ in phones), 2)
            self.assertLess(len(phones), len(word) + 3)
            self.assertEqual(sum(stress == 1 for _, stress in phones), 1)
        self.assertEqual(self.phonemizer.word("josiana's")[0][-1], ("z", 0))

    def test_names_in_leigh_and_words_in_virus_are_put_together_from_their_parts(self):
        word, text = self.phonemizer.word, lambda spelling: convert.to_text(self.phonemizer.word(spelling)[0])
        self.assertEqual(word("ryleigh"), word("ryley"))                    # the dictionary's "Ryley"
        self.assertEqual(word("kayleigh"), word("kayley"))                  # the model, on the spelling it reads well
        self.assertEqual((text("ryleigh"), text("kayleigh's"), text("paisleighs")), ("ɹ ˈaɪ l i", "k ˈeɪ l i z", "p ˈeɪ z l i z"))
        self.assertEqual((text("leigh"), text("sleigh"), text("raleigh")), ("l ˈiː", "s l ˈeɪ", "ɹ ˈɔː l i"))     # dictionary words stay
        self.assertEqual((text("coronavirus"), word("coronavirus")[1]), ("k ɚ ɹ ˈoʊ n ə v ˌaɪ ɹ ə s", "ending"))
        self.assertEqual(text("coronaviruses"), "k ɚ ɹ ˈoʊ n ə v ˌaɪ ɹ ə s ᵻ z")
        self.assertEqual(text("coronavirus's"), "k ɚ ɹ ˈoʊ n ə v ˌaɪ ɹ ə s ᵻ z")
        self.assertEqual(word("norovirus")[1], "model")                     # the first part is a guess, "virus" is not
        self.assertTrue(text("norovirus").endswith("v ˌaɪ ɹ ə s"))
        self.assertEqual((word("virus")[1], word("antivirus")[1]), ("dictionary", "dictionary"))
        self.assertEqual((text("bobsleigh"), text("bobsleighs")), ("b ˈɑː b s l eɪ", "b ˈɑː b s l eɪ z"))      # a sleigh, not a name
        self.assertEqual((text("provirus"), text("proviruses")), ("p ɹ oʊ v ˈaɪ ɹ ə s", "p ɹ oʊ v ˈaɪ ɹ ə s ᵻ z"))
        engine = Phonemizer(BUILT, overrides={"emmaleigh": "EH1 M AH0", "lyssavirus": "L IH1 S AH0"})
        self.assertEqual(self.speak("Emmaleigh, lyssavirus", engine), "ˈɛ m ə , l ˈɪ s ə")      # the user's own words still win

    def test_sounds_people_write_and_the_letters_that_stay_letters(self):
        self.assertEqual(self.speak("Sh! Ss! Tsk!"), "ʃ ! s ! t ˈɪ s k !")
        self.assertEqual(self.speak("Sh Ss Tsk"), "ʃ <sp> s <sp> t ˈɪ s k")
        self.assertEqual(self.speak("Sss. Zzzz. Mmmm. Pssst."), "s . z . m . p s t .")
        # "ss" beside a word, in capitals or inside a longer word is two letters, as the dictionary has it
        self.assertEqual(self.speak("The SS and ZZ Top"), "ð ɪ <sp> ɛ s ˈɛ s <sp> æ n d <sp> z ˌiː z ˈiː <sp> t ˈɑː p")
        self.assertIn("ɛ s ˈɛ s d ˌiː ˈɛ s", self.speak("Two SSDs"))
        self.assertIn("m : ɛ s ˈɛ s", self.speak("The format is HH:mm:ss."))
        self.assertIn("ð ɪ <sp> ɛ s ˈɛ s <sp> ˈɛ n t ɚ p ɹ ˌaɪ z", self.speak("He boarded the ss enterprise."))
        self.assertEqual(self.speak("THE SS MEN SAID TSK AND PSST"), "ð ɪ <sp> ɛ s ˈɛ s <sp> m ˈɛ n <sp> s ˈɛ d <sp> t ˈɪ s k <sp> æ n d <sp> p s t")

    def test_a_hyphen_before_read_decides_the_past_form(self):
        choose = self.phonemizer.homographs.choose
        self.assertEqual(choose("read", ["She's", "a", "well", "read", "girl", "."], 3, joined=True)["id"], "past")
        self.assertIn("w ˈɛ l ɹ ˈɛ d", self.speak("Well-read people read a lot."))
        self.assertIn("p ˈiː p əl <sp> ɹ ˈiː d", self.speak("Well-read people read a lot."))

    def test_phrase_table_can_be_switched_off(self):
        joined = self.speak("Most of the time")
        apart = self.speak("Most of the time", Phonemizer(BUILT, join_phrases=False))
        self.assertIn("ʌ v ð ə", joined)
        self.assertIn("ʌ v <sp> ð ə", apart)
        self.assertEqual(apart.count(SP), joined.count(SP) + 1)

    def test_identity_is_stable_and_tracks_the_build(self):
        first = phonemizer.identity(BUILT)
        self.assertEqual(first, phonemizer.identity(BUILT))
        self.assertEqual(set(first["built"]), {"lexicon.tsv", "align.json", "g2p.json", "g2p.npz", "homographs.json"})
        self.assertLessEqual(self.phonemizer.model.parameters, 462_709)     # headroom of the TTS model under 5M


def phonemizer_variable():
    from dizione.engine import ENVIRONMENT
    return ENVIRONMENT


if __name__ == "__main__":
    unittest.main()
