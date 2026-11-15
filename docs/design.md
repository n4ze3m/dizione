# dizione: design notes

How dizione turns a sentence into phonemes, what was measured, and what is
still wrong. It began as the front end of the tiny-voca text-to-speech model,
written to replace eSpeak NG so that the whole system could be released under a
permissive licence and run in a browser or on a phone. Sections that compare
against eSpeak or mention annotations and checkpoints refer to that project.

The output is 76 symbols: 59 phones, stress as a separate 0/1/2 feature,
`<bos>`, `<eos>`, `<sp>` and punctuation.

## Using it

```bash
export DIZIONE_DATA=/path/to/built        # see "Building"
python -m dizione --words "Hello Samir, it costs \$3.50."
```

In code:

```python
import dizione
result = dizione.phonemize("Hello Samir")  # symbols, ids, stress, unknown
```

**Your own pronunciations** come first, before anything else:

```python
from dizione.engine import Phonemizer
p = Phonemizer(overrides={"Samir": "S AH0 M IH1 R", "kubernetes": "k ˌuː b ɚ n ˈɛ ɾ iː z"})
```

or a file named by `DIZIONE_OVERRIDES` (one word per line, two spaces, the
pronunciation). A pronunciation is either CMUdict notation (`HH AH0 L OW1`;
the conversion rules below are applied) or phones of the inventory with stress
marks (`h ə l ˈoʊ`; used as written).

**Inline, for one word in one text:** `Say {HH AH0 L OW1} to Samir{S AH0 M IH1 R}.`
Braces alone are spoken as given; braces right after a word replace that word.

## Teaching it new words

New words appear faster than dictionaries: names, slang, memes. Three places
take a pronunciation, from the most to the least lasting:

1. **For everyone:** a line in `src/dizione/data/exceptions.txt`
   (word, two spaces, pronunciation). 219 words are there now, among them 55
   of today's slang ("skibidi", "rizz", "delulu", "lowkey", "bussin"), 43
   meme names ("tralalero", "bombardiro", "sahur") and 6 sounds people write
   ("tsk", "psst", "pfft"). No rebuild is needed.
2. **For you:** your own file, named by `DIZIONE_OVERRIDES`, or
   `Phonemizer(overrides={...})`. Same format; it wins over the built-in list.
3. **For one sentence:** braces in the text, as above.

**A name of several words** can have its own rhythm. Write one pronunciation
per word, separated by `|`; the stress you write is the stress spoken:

```
tung          T UW1 NG
sahur         S AA0 HH UH1 R
tung tung tung sahur  T UW1 NG | T UW2 NG | T UW2 NG | S AA0 HH UH1 R
```

"Tung Tung Tung Sahur" then comes out as `t ˈuː ŋ <sp> t ˌuː ŋ <sp> t ˌuː ŋ
<sp> s ɑː h ˈʊɹ`: TOONG-toong-toong sah-HOOR. The single words keep working on
their own ("Tung, sahur!"). The same form works in your own file and in
`overrides={"skibidi toilet": "S K IH2 B IH0 D IY0 | T OY1 L AH0 T"}`.

**Writing a pronunciation** in CMUdict notation, from the way a word sounds:

| You hear | Write | You hear | Write | You hear | Write |
|---|---|---|---|---|---|
| ah (father) | `AA` | ee (see) | `IY` | oo (boot) | `UW` |
| a (cat) | `AE` | i (sit) | `IH` | oo (book) | `UH` |
| uh (cut, sofa) | `AH` | oh (go) | `OW` | ur (her) | `ER` |
| aw (law) | `AO` | oy (boy) | `OY` | ow (cow) | `AW` |
| ay (day) | `EY` | eye (my) | `AY` | e (bed) | `EH` |

Every vowel ends in a digit: `1` for the loudest syllable, `2` for a lesser
stress, `0` for none ("sah-HOOR" is `S AA0 HH UH1 R`). Consonants are
`B CH D DH F G HH JH K L M N NG P R S SH T TH V W Y Z ZH`, where `DH` is the
"th" of "this", `TH` of "thin", `JH` of "joy", `ZH` of "vision". Check a word
with `python -m dizione --words "Tung Tung Tung Sahur"`.

Chat abbreviations that stand for words ("ngl", "fr", "istg") or are read as
letters ("pov", "afk") go in `data/chat.txt` instead. A row marked "alone" is
read that way only as a reply by itself: "k" is "okay" in "k", "k, thanks" and
"see you there, k?", and the letter K in "the value of k", "8 k" and "k = 5".

The pronunciations in these tables were typed by hand for this project; none
comes from eSpeak, misaki or Wiktionary.

## What happens to a sentence

| Step | File | What it does |
|---|---|---|
| 1. Text rules | `normalize.py`, `data/abbreviations.txt`, `units.txt`, `chat.txt`, `endings.txt` | Numbers, ordinals, years, money, dates, times, telephone numbers, units, abbreviations, initialisms, symbols, versions, codes, file names, web and mail addresses become words (see "Text rules"). Emoticons and emoji are dropped: never read, never a pause. Chat forms are expanded (`btw`) or spelled (`lol` is read "L O L"). Accents are removed (`café`), stretched words undone (`soooo`). |
| 2. Your overrides, inline phonemes, taught names | `engine.py`, `data/exceptions.txt` | Win over everything. |
| 3. Function words and groups | `sentence.py`, `data/function_words.txt`, `data/phrases.txt` | 160 function words have a weak form and a stress class; 49 pairs such as "of the" are spoken as one word. |
| 4. Homographs | `homographs.py`, `data/homograph_rules.txt` | "read", "live", "use", "record": hand rules for a few grammar-driven cases, then a linear classifier on the neighbouring words. |
| 5. Dictionary | `convert.py`, `align.py`, `data/exceptions.txt` | 124,926 words from CMUdict, converted to the TTS phones by rule when the data is built. 219 hand-written exceptions (recent words, slang, names, sounds, usual pronunciations), one of them a name of several words. |
| 6. Word endings | `morphology.py` | Plurals, possessives, -ed, -ing, -ly and other regular endings, some prefixes, compounds of two dictionary words, British spellings. |
| 7. Words put together from parts | `engine.py` | Two kinds of new word the model reads badly. A name in "-leigh" is said like the same name in "-ley" ("Kayleigh", "Ryleigh"). A word in "-virus" is its first part, from the dictionary or the model, and then "virus" ("coronavirus"). |
| 8. Unknown-word model | `g2p.py` | Letters to phonemes in one forward pass (NumPy). |
| 9. Sentence rules | `sentence.py` | Sentence stress, "the" and "to" before a vowel, linking r, the flap of "it" and "at" before a vowel. |

**What the rules add to CMUdict.** CMUdict writes 39 broad phonemes. The model
was trained on a narrower transcription, and `convert.py` produces it:

* the flap in "water", "city", "party" (`ɾ`), the glottal stop and syllabic n in
  "button", "certain" (`ʔ n̩`), syllabic l in "little" (`əl`);
* r-coloured vowels as one phone (`ɑːɹ ɔːɹ oːɹ ɪɹ ɛɹ ʊɹ aɪɚ`), split again before
  a vowel ("story" is `s t ˈoː ɹ i`), and `ɚ`/`ɜː` with a linking `ɹ`;
* five reduced vowels where CMUdict has two: `ɐ` for an initial "a" ("about"),
  `ᵻ` in "-ed"/"-es" and weak prefixes ("wanted", "before"), `i` at the end of
  "happy", `ɪ` or `ə` by spelling ("cabin" against "sofa");
* "more" against "for" (`oːɹ`/`ɔːɹ`) and "long" against "law" (`ɔ`/`ɔː`), which
  CMUdict merges, decided by spelling;
* `ŋ` before k and g ("income"), no secondary stress next to the primary one.

Many of these depend on how a vowel is written, so every dictionary entry is
first aligned to its spelling ("thought" is th|ough|t). The alignment table is
estimated from CMUdict alone, by expectation-maximisation.

## Text rules

What `normalize.py` says for things that are not words. `[A B]` means letter
names; a comma is a short break.

| Written | Said |
|---|---|
| `v2.4.1`, `Version 10.0.19045`, `6.5.0-14` | "V two point four point one"; "ten point zero point one nine zero four five" (a part of three digits or more is read digit by digit); "six point five point zero dash fourteen" |
| `Python 3.12`, `Ubuntu 24.04`, `v3.10` | after "version", "release", "firmware", "update" and common product names the part after the point is a whole number: "three point twelve", "twenty four point oh four". Elsewhere `3.12` stays "three point one two" |
| `gradium.sh/docs`, `sales_2026-Q3.xlsx`, `help@shop.example.co.uk` | "gradium dot S H slash docs"; "sales underscore twenty twenty six dash Q three dot X L S X"; "help at shop dot example dot co dot U K". `data/endings.txt` lists the endings said as words (com, org, json, zip, log); every other ending is spelled |
| `the end.Then` | a missing space, not a file: an unlisted ending counts only when the name has a slash or an underscore |
| `on 14/03`, `Delivered on 12/10`, `due 31/12`, `01/07 to 30/09` | "March fourteenth", "December tenth", "December thirty first", "July first to September thirtieth". A date when a part has a leading zero, when the first number is over 12, or after a date word (on, due, until, shipped, a weekday). Month first, unless a date in the same text shows the day first |
| `3/4`, `7/10`, `24/7`, `50/50` | "three quarters", "seven tenths", "twenty four seven", "fifty fifty", as before |
| `15.11.2026`, `3-17 November`, `22 February.` | "November fifteenth twenty twenty six"; "the third to the seventeenth of November"; the full stop after a month name ends the sentence again |
| `1½`, `2¾`, `2 1/2`, `½` | "one and a half", "two and three quarters", "two and a half", "one half" |
| `≥ ≤ ≠ ≈ ∞ √ → ^`, `>=`, `7 − 3`, `120 m²` | "greater than or equal to", "less than or equal to", "not equal to", "approximately", "infinity", "square root of", "to", "to the power of"; "seven minus three"; "one hundred twenty square meters" |
| `67 > 66`, `see if b>a`, `x<5`, `p<.05`, `a >= b`, `0 < x < 1` | a comparison between two values is said with "is": "sixty seven is greater than sixty six", "see if B is greater than A", "X is less than five", "A is greater than or equal to B", "zero is less than X is less than one". A value is a figure, a single letter or a closing bracket, with the same space (or none) on both sides of the sign. No "is" where the comparison only describes the value ("for n > 4", "any q > 0", "an integer k ≥ 2") or where a word stands on a side ("values > 10", "a >50% rise"): these say "greater than" as before |
| `b>a`, `x<5`, `File>Open`, `<b>`, `map<int, int>`, `you<3` | "<" and ">" with no space are read when a figure stands on one side or a single letter or bracket on both. A menu path, a name ("M>Train"), a tag and the brackets of a type are not comparisons; "<3" after a word is a heart and is not read |
| `a=10`, `c = a + b`, `(a)`, `f(a, b)`, `A/B`, `k = k + 1` | a letter beside a sign or alone in brackets is named: "A equals ten", "C equals A plus B". This holds for "A", "a" and the letters chat writes for a word ("k", "u", "r"); in "mode = a set of notes" and "a +1" the article stays |
| `y=(a+b)/2`, `10-3=7` | "Y equals ( A plus B ) two": "=(" is a sad face only where no value follows it; "ten minus three equals seven" |
| `iOS`, `macOS`, `iPadOS`, `eSIM` | "I O S", "mac O S", "I pad O S", "E sim". `iPhone`, `iPad`, `eBay`, `YouTube` are dictionary words |
| `#A7-4419`, `KT-738-RP`, `AB-22`, `KX-300`, `gate B12`, `seat 14C`, `Pixel 8a` | "number A seven, four four one nine"; "K T, seven three eight, R P"; "A B twenty two"; "K X three hundred"; "B twelve"; "fourteen C"; "eight A". A letter next to a digit or inside a code is always named |
| `SAVE-20`, `COVID-19`, `5-year`, `20-year-old`, `10-15`, `2019-2021` | unchanged: "save twenty", "covid nineteen", "five year", ranges with "to" |
| `4502 1187 3390 2`, `91-220-0017`, `CS-2026-0813`, `2026/0917/44`, `RR 123 456 789 GB` | digit by digit with a break between groups: "four five zero two, one one eight seven, three three nine zero, two" |
| `Order 48213907`, `PIN is 4471`, `ref. no. is 4471`, `error code 404` | digit by digit: a number of seven digits or more that is not round, and a shorter one after "number", "ID", "reference", "PIN", "code", "extension". `1500000 people` and `1 000 000` stay amounts |
| `020 7946 0018`, `+44 20 7946 0018`, `call 999`, `text 60066` | telephone numbers say "oh": "oh two oh, seven nine four six, oh oh one eight"; "plus four four, two oh, ..." |
| `Bug #2093`, `Order #A7`, `#1`, `#winning` | "number two zero nine three", "number A seven", "number one", "hashtag winning" |
| `2:03:45`, `48:12 long`, `at 10:30:15 AM` | "two hours, three minutes and forty five seconds"; "forty eight minutes and twelve seconds"; after "at" or with AM/PM it stays a time of day |
| `20:00`, `14:30`, `14:00-16:00`, `7.30 pm`, `4 p.m. ET` | "twenty hundred", "fourteen thirty", "fourteen hundred to sixteen hundred", "seven thirty P M", "four P M E T" |
| `16:9`, `3:1`, `1:25,000` | "sixteen to nine", "three to one", "one to twenty five thousand" |
| `-€12.50`, `50¢`, `CHF 28` | "minus twelve euros and fifty cents", "fifty cents", "twenty eight Swiss francs" |
| `20 €`, `20€`, `100$`, `5-10 €`, `-3 €`, `750 000 $` | a currency sign after the amount: "twenty euros" (twice), "one hundred dollars", "five to ten euros", "minus three euros", "seven hundred fifty thousand dollars" (before such a sign, spaces may part the thousands). The second sign of a formula is not one: `$x = 2$` is "X equals two" |
| `Ss!`, `Sss`, `Zzzz`, `Sh, ss, tsk` | a hiss and a snore: a letter s or z typed three times or more, or twice when it stands alone or among other sounds. In `mm:ss`, `the ss`, `SS` and `SSDs` the two letters are letters |
| `1.299,50`, `€1.299,50`, `1.299,50 €`, `12,50 EUR`, `3,5%`, `1,5 kg` | figures written the continental way: "one thousand two hundred ninety nine point five zero"; "one thousand two hundred ninety nine euros and fifty cents" (twice); "twelve euros and fifty cents"; "three point five percent"; "one point five kilograms". A comma is read as a decimal comma only where the form leaves no doubt: after thousands parted by points, next to a currency or a percent sign, or before a unit. `1,2 and 3` stays a list and `2.500` stays "two point five zero zero" |
| `3.5 m`, `1.5 L`, `180 hp`, `2 tsp`, `5,000 mAh`, `30 min. by bus` | meters, liters, horsepower, teaspoons, milliamp hours; the stop after a unit is not a sentence end before a small letter. One-letter units are read in one letter case only (`data/units.txt`): `5G` and `5 M` are not units |
| `3 × 4 metres`, `2 m × 1 m` | "three by four metres", "two meters by one meter" |
| `Mon.-Fri.`, `max. load`, `Est. arrival`, `3rd fl.`, `Sizes S-XL` | "Monday to Friday", "maximum load", "estimated arrival", "third floor", "S to X L" |
| `RTX 4070 Ti`, `ISO 9001` | "R T X forty seventy T I", "ninety oh one": four digits after a name in capitals are read in pairs |

## Sources and licences

| Used for | Source | Licence |
|---|---|---|
| Dictionary, alignment table, unknown-word model | CMUdict, commit `74790861` (25 Oct 2025) | BSD-style (Carnegie Mellon University); keep its copyright notice with the built dictionary |
| Homograph classifier weights | Wikipedia Homograph Data, commit `8f008f02` | Apache-2.0 (Google). The sentences are from Wikipedia; only the trained weights are shipped, never the sentences |
| Everything in `data/*.txt` | written by hand for this project | the project's (Apache-2.0, like the code) |

Nothing generated by eSpeak is in the dictionary, the tables or the model, and
no misaki or Wiktionary data was used. eSpeak was installed in the build
sandbox as a yardstick only: its tokens for the corpus sentences are the
reference of the agreement report below.

One thing to know about the hand-written tables. The phone inventory and its
conventions were eSpeak's to begin with, and the model learned them. So where
English allows several weak forms ("of" with `ʌ` or `ə`), and for which
function-word pairs are joined, the tables follow the tokens the model was
trained on; that was decided by looking at the agreement report. They are a
few hundred short entries of ordinary English pronunciation, typed by hand,
not copied from eSpeak's files. `Phonemizer(join_phrases=False)` switches the
pair table off if you would rather not carry that habit over.

## Measurements

All numbers were measured on 14 Nov 2026.

### Unknown-word model

Letters in, CMUdict phonemes with stress out. An embedding per letter (two
output positions per letter), two bidirectional GRU layers of 127 units, a CTC
output layer; one forward pass, greedy decoding. **458,682 parameters**; the
TTS model has 462,709 to spare under its 5 million limit, so it fits with
4,027 left and nothing has to be trimmed (4,995,973 together). 1.8 MB in
float32.

Trained on CMUdict only. Words are split by a hash of the spelling: 115,085
words for training, 2,546 for choosing the checkpoint, 6,235 held out. 2,186
abbreviation-like entries ("dept", "fbi") are excluded from all three.

| Held-out CMUdict words (6,235) | Word error | Phoneme error |
|---|---:|---:|
| phonemes and stress | 32.4% | 7.9% |
| phonemes only | 26.5% | 6.1% |

A word counts as right when the output equals any pronunciation CMUdict lists
for it (against the first one only: 34.4%). So two unseen words in three come
out exactly as in the dictionary, and a wrong word has one or two wrong
phonemes. On the 2,546 words used to choose the checkpoint: 32.1% and 27.0%. For
orientation, the LiteG2P paper (the reference design) reports 26.5% word error
at 600,000 parameters and 24.3% at 1.27 million on CMUdict. Its test split
and its handling of stress are not the ones used here, so the figures are not
directly comparable.

What mattered, in fifteen short runs (twelve of them on a rented GPU, about
15 minutes each; their logs are kept with the reports): dropout and weight
decay. With the usual dropout of 0.15 the model under-fits (36.6% word
error after 100 epochs, the same as after 30); with none it reaches 34.5 to
35.4%; with none and a weight decay of 0.05, 32.4%. Convolutions before the
GRU layers made no difference. A running average of the weights is kept and
the better of the two is saved each epoch. The checkpoint was chosen on the
validation words; the held-out words were only scored afterwards.

In the corpus only 0.6% of the words reach the model (400 of 66,635); another
0.5% are built by the ending rules.

### Homographs

162 homographs, 326 senses. Pronunciations: 264 senses matched to a CMUdict
entry, 48 written by hand (`data/homograph_fixes.txt`), 14 taken from the
source's own transcription.

| Held-out sentences (1,615) | Accuracy | Always the commonest sense |
|---|---:|---:|
| all 162 words | 95.4% | 84.0% |
| 128 noun-or-verb words (record, present, use) | 97.0% | 84.0% |
| 34 other words (read, lead, bass, bow) | 89.1% | 84.3% |

The 128 noun-or-verb words share one set of weights on the grammatical class
of their neighbours, so the verb reading is found even for a word whose verb
sense never occurs in the training sentences. 855 non-zero weights in total.
The training sentences are from Wikipedia, where questions, commands and "I"
are rare; 15 hand rules cover the cases that matter in speech ("have you
read" / "will you read", "take a bow"). They add 0.4 points on the held-out
set and fix every such sentence in `tests/data/tricky_sentences.txt`. A
sixteenth rule came after the release and is not in that figure: "read"
joined by a hyphen to "well", "widely", "best" and the like is the past form
("a well-read girl").

### Agreement with eSpeak on the corpus

4,799 distinct sentences, 314,489 tokens, 237,234 phones. Edit distance
between the two token sequences, divided by the length of eSpeak's.

| | With the pair table (default) | Without |
|---|---:|---:|
| all tokens (phones, word boundaries, punctuation) | 98.80% | 97.88% |
| phones | 98.48% | 98.05% |
| phones with their stress | 98.17% | 97.52% |
| stress, on the vowels both agree on | 99.19% | 98.59% |
| sentences with identical tokens and stress | 61.1% | 45.4% |
| sentences with identical phones | 65.2% | 57.3% |

Plain prose (4,674 sentences): 98.68% of phones, 62.2% of sentences identical.
Sentences with digits or symbols (125): 91.73%, on purpose (see below).
These are the numbers of the second pass; the first pass had 98.81% of all
tokens and 91.93% on the digit sentences, everything else the same to the
last digit shown.

**Where they differ, and why** (phones, plain prose only; 3,731 differences in
230,245 phones):

| Kind of difference | Phones | Why |
|---|---:|---|
| Which weak vowel (`ə ɐ ɪ ᵻ i`) | 1,472 | eSpeak picks the reduced vowel of each word by its own spelling rules ("secret" `ᵻ`, "pocket" `ɪ`, "cousin" `ə`); ours follow a dozen general rules. All are unstressed and close in sound. |
| Stress only | 694 | CMUdict and eSpeak mark secondary stress differently ("indeed", "conversation"); eSpeak stresses the last syllable of "-ative", "-ist" words; a few function words in positions the classes do not cover ("is" at the end of a clause). |
| A phone more or less | 490 | Different pronunciations of the same word ("handsome" with or without d, "valuable" with three or four syllables), unknown words. |
| Another vowel | 287 | Mostly unknown words and names, where both guess. |
| Vowel against consonant | 264 | Syllabic l written `əl` by one and `ə l` or `l` by the other ("trembling", "doubtfully"). |
| Consonants | 182 | s/z ("husband"), names. |
| `ɑː ɔ ɔː` | 144 | eSpeak is not consistent here ("dog" and "office" with `ɑː`, "log" and "off" with `ɔ`); ours follows CMUdict's vowel and the spelling. |
| `ɔːɹ` against `oːɹ` | 105 | eSpeak says "storm", "according" with the vowel of "more"; ours uses the historical word classes. American speech has one vowel for both. |
| t, flap, glottal stop | 64 | "pattern", names in "-ton". |
| r | 29 | Linking r after "Mr." and before a few words. |

Counted by where a word's pronunciation came from: function words differ from
eSpeak in 0.6% of cases, joined pairs in 1.3%, dictionary words in 6.6%
(mostly one weak vowel), homographs in 14%, words built by ending rules in
27%, unknown words in 66%.

**Differences that are intended:**

* Numbers. eSpeak reads "1681" as "one thousand six hundred eighty one" and
  "12/09/2026" digit by digit; ours says "sixteen eighty one" and "December
  ninth, twenty twenty six", as a person (and the voice that recorded the
  corpus) would. The annotation notes already list these rows as the ones
  where eSpeak and the audio disagree.
* Homographs. eSpeak has one pronunciation for "read", "live", "use",
  "close", "tears"; ours chooses. 14% of homograph tokens differ.
* Word boundaries. With the pair table off, every boundary gets a token.

### Speed

One CPU core, Python: 0.1 s to load, 4,400 sentences (334,000 characters) per
second, 0.9 ms per unknown word. The first pass, timed on the same machine in
the same hour, did 4,600; text without a digit skips the number rules, which
is why the new rules cost 5% and not 40%. (The first write-up said 3,700, from
an earlier session; both versions were timed again here with one script.)
The package needs only NumPy.

### Hard sentences

The 500 English rows of `mrfakename/tts-hard-sentences-202610` (CC BY 4.0;
used as a test only, downloaded in the sandbox, not copied into this
repository): 20 kinds of sentence, 25 each. Every sentence was run through
the normaliser and the words it produced were read, all 500, before and
after the second pass.

206 of the 500 are read differently now. By kind: codes 23 of 25, order
numbers 24, telephone numbers 24, file names and addresses 21, mixed fields
18, versions 14, symbols 12, short replies 11, units 10, times 9, number
signs 8, dates in words 8, dates with slashes 8, abbreviations 7, ranges 4,
currency 3, percents and fractions 2; ordinals, quantities and temperatures
were right already and did not change. No sentence fails and none produces a
symbol outside the inventory. What is still not as a careful reader would say
it is listed under "What is still wrong", points 6 and 7.

## The second pass: what changed and what did not

For whoever re-annotates the corpus.

* **The front-end fingerprint did not change.** `text.frontend_fingerprint()`
  for `own` is still `57b0550a…d74a8a`: it covers the version number (4) and
  the 76 symbols, and neither moved. eSpeak's is still `a6f1cf32…be338`.
* **The build fingerprint did change**, as it does with any edit:
  `phonemizer.identity()["fingerprint"]` went from `d814e2e6…ec0593c` to
  `e2f0c0dc…ec78dc12`. An annotation run records it, so a run started with
  the first pass cannot be resumed with this one; start it with this one.
* **The built data did not change.** The dictionary, the alignment table, the
  model and the homograph weights were rebuilt with the new code and are the
  same bytes as the files in the model repository.
* **Plain prose: one sentence in 4,674 changed.** Every corpus sentence was
  phonemized with the old and the new code and the tokens and stress compared.
  The one is "The keyboard shortcut is Ctrl, Shift, and the letter R
  together": "Ctrl" was an unknown word for the model to guess and is now
  "control". The other 4,673 are identical, token for token.
* **Sentences with digits or symbols: two in 125 changed**, both meant:
  "error code 404" is "four zero four" and "area code 617" is "six one seven".
* **Prose can still differ outside the corpus** in exactly these cases: a word
  that is now taught (the slang and names in `exceptions.txt`, the chat forms
  "fr", "ngl", "pov" and the like, "Ctrl", "Esc"); `iOS`-style words (small
  letters, then capitals); a range of days or months with a hyphen
  ("Mon-Fri", "May-June" is "May to June"); "max.", "min.", "est." before a
  small letter; "ref.", "fl.", "ct." and the other new abbreviations; "Fr."
  before a name ("Father"); a sentence that starts with "A" right after a
  word like "plan" or "part" ("a plan. A dog" was the letter A, a bug); and
  the names "Tung", "Vaca", "Bae", "Pookie", whose dictionary entries were
  replaced. Apart from "Ctrl", none of these occurs in the corpus; the
  comparison above is the proof.

## After the release: what changed and what did not

Fixes for words and figures people reported, made on 15 Nov 2026. For whoever
uses the tokens of an earlier build.

* **The front-end fingerprint did not change.** tiny-voca's
  `text.frontend_fingerprint()` is still `c729f235…7b69d4da`: version 4 and
  the 76 symbols are as they were. The released checkpoint loads and speaks
  with this build.
* **The build fingerprint did change**, as with any edit to the source.
  `identity()["fingerprint"]` was `35f4abab…33bbb39c` in the release.
* **The built data did not change.** The dictionary and the homograph weights
  were rebuilt with the new code and are the same bytes as the released files.
* **The corpus did not change.** All 8,749 distinct sentences of the dataset
  were phonemized with the old and the new code: tokens and stress are
  identical in every one. So are 164 of the 166 sentences the test file had
  before; the other two hold a comparison between two values and now say
  "is" ("x = 5 and y < 10", "where x ≠ y").
* **Prose can differ outside the corpus** in exactly these cases:
  the words now taught ("Emmaleigh", "Hermione", "Persephone", "coeliac",
  "celiac", "scabies", "lyssavirus", "provirus", "bobsleigh"); a word in
  "-leigh" or in "-virus" that the dictionary and the ending rules do not
  know; "read" after a hyphen and one of the fifteen words of the new rule;
  the sounds "tsk", "psst", "pfft", "hmph", "psh", "pssh" in small letters or
  in capitals; a hiss and a snore ("Sss", "Zzzz", and "Ss" or "Zz" standing
  alone or among other sounds); and "mmm", which is the hum "mm".
* **Figures differ** where a comma is now a decimal comma (see "Text rules"),
  where a currency sign follows the amount (it was dropped: "20 €" was
  "twenty"), where a minus stands before such an amount (it was dropped before
  a currency code), and in a range before a sign ("5-10 EUR" was "five ten
  euros").

The first version of these fixes was then checked on text it had never seen:
1,985,590 sentences of news, film reviews, tweets, Simple English Wikipedia
and chat with an assistant, 1,322,304 forms of 440,455 words and names, and
22,187 figures in every notation, each read by the released code and by the new
code. That check found three mistakes, all corrected:

* "ss" and "zz" had been taught as words, a hiss and a snore. But "ss" is two
  letters far more often: "SSDs", "VHSs", "HH:mm:ss", "the ss", a variable
  named `ss`, the German "ß". They are letters again, exactly as released, and
  only the text rules above make a hiss or a snore.
* A figure before a dollar sign was read as a price also at the end of a
  formula: "$x = 2$" said "two dollars". Formulas are read as before.
* "750 000 $" lost its thousands ("seven hundred fifty zero dollars").

After the corrections 788 of the 1,985,590 sentences are read differently
from the release, and 300 of the 1,322,304 word forms. No sentence, word or
figure that the released code could read fails with the new code. The
sentences by cause:

| Cause | Sentences | Example |
|---|---|---|
| A word in "-virus" | 377 | "coronavirus": `k ɚ ɹ ˌɑː n ə v ˈaɪ ɹ ə s` became `k ɚ ɹ ˈoʊ n ə v ˌaɪ ɹ ə s` |
| A word now taught | 174 | "Hermione", "Persephone", "tsk", "celiac", "psst", "bobsleigh" |
| A currency sign after the amount | 64 | "12$", "27 $", "800 €", "750 000 $" |
| A decimal comma | 62 | "9,9°C", "85,44 km²", "12,5%", "CHF 2,70 billion" |
| A hiss, a snore, a hum | 52 | "Zzzz", "Mmm", "Sssssss" |
| A name in "-leigh" | 48 | "Rayleigh", "Eastleigh" |
| A taught sound written long | 7 | "Pfffft", "psssst" |
| "read" after a hyphen | 4 | "well-read" |

One of the 64 is wrong: a formula cut in two by a sentence break ends in
"n_0=1$", with no sign before it to show a formula, and says "one dollar".

### Signs without spaces, a lone "k", and "is"

Three more things were reported and changed on the same day.

* **A sign with no space around it lost words.** "let a=10, b=20, see if b>a"
  was read "let a equals ten, B equals twenty, see if B a": the ">" was
  dropped and both "a" were the weak article, so the sentence seemed to break
  off. It is now "let A equals ten, B equals twenty, see if B is greater than
  A", the same as with spaces. The voice and the demo were not at fault:
  every phoneme got its frames, and the text went to the voice in one piece.
  At speed 0.70 the audio grew from 4.66 to 6.13 seconds.
* **A lone "k" was always "okay".** It is the letter K now, and "okay" only
  as a reply by itself (see `data/chat.txt`).
* **A comparison between two values says "is"** ("sixty seven is greater than
  sixty six"). This was asked for; it changes what the release said.

The same three sets were read again. Of the 1,985,590 sentences this round
changes 1,900, none of which fails; no word form changes.

| Cause | Sentences | Example |
|---|---|---|
| A letter beside a sign or alone in brackets is named | 939 | "Side A = 8 cm", "gcd(a, b)", "(a) it was an Italian film", "A/C" |
| A lone "k" is the letter, not "okay" | 563 | "the value of k", "401(k)", "k-means", "1.4k years" |
| "Is" in a comparison between two values | 293 | "while n > 1", "when m < n", "as 4<12" |
| "=(" is not a sad face | 51 | "A=(1,2)", "figsize=(12, 12)" |
| "<" or ">" with no space, that was dropped, is read | 30 | "if A>B and B>C, then A>C", "if n<0" |
| A heart joined to a word is not read | 17 | "tomorrow<3" said "tomorrow three" |
| Minus, not "to", before "=" | 7 | "66-4=62" |

Together with the table above, 2,688 of the 1,985,590 sentences and 300 of
the 1,322,304 word forms are now read differently from the release.

## Tests

`tests/test_dizione.py` (42 tests) and `tests/data/tricky_sentences.txt`
(223 sentences: 187 with the words they must turn into, 84 stretches of phones
that must appear). Every bug fixed has a sentence there.
Run them with `PYTHONPATH=src python -m unittest discover -s tests`.

* Without the built data: the conversion rules on 40 aligned words, the
  aligner, number and date rules, emoticons, inline phonemes, sentence stress
  and linking, word endings, every hand-written table, the symbol inventory,
  and the NumPy model against PyTorch.
* With the built data (`DIZIONE_DATA`): every tricky sentence, a greeting,
  overrides, unknown words, the pair table switch, the identity.

30,000 random strings of digits, letters and signs go through without an
error.

## What is still wrong

1. **Unknown words.** One in three has an error (one in four if stress is
   not counted), usually one vowel or the stress. Foreign names are the worst ("Circe", "Phaeton"). Add names you
   care about to the override list. British spellings with "oe" are guessed
   too and come out wrong ("oesophagus", "oestrogen", "manoeuvre"); only
   "coeliac" is taught. A rule that writes "e" for "oe" was checked against
   the dictionary and dropped: it turns German names into other names
   ("Boehm" into "Behm").
2. **Homographs outside the 162** are not disambiguated ("does" as the plural
   of "doe", "bows", "moped"). The classifier's sense of what is common comes
   from Wikipedia; the 34 words that are not noun-or-verb pairs are right 89%
   of the time on held-out sentences. A hyphen decides "read" only after the
   words of its rule: "re-read" and "well read" without a hyphen are left to
   the classifier.
3. **Not heard yet.** The current model has never spoken these tokens. The
   agreement numbers say they are close to what it knows; the proof is the
   fine-tune and a listening test.
4. **Sentence stress is by word class only.** No emphasis, no contrast, no
   difference between "is" at the end of "what it is" and of "there it is".
5. **Weak vowels.** Where eSpeak and ours disagree (0.6% of phones) neither is
   wrong, but the model has heard eSpeak's choice for that word.
6. **Numbers and dates are American and guessed from form.** "12/09" is
   December 9th and "09/10" September 10th unless another date in the same
   text shows the day first; a British reader means 9 October. The month-first
   order is kept on purpose (decided 14 Nov 2026); it is listed here so that
   it surprises nobody. A four-digit
   number from 1100 to 1999 is read as a year; 2000 to 2099 is a year unless a
   plural noun follows. Whether a number is a reference (digit by digit) or an
   amount is decided by the words before it and by its length: "order 15000
   units" is an amount, "a population of 8336817" is wrongly read digit by
   digit, "ticket 4471" without "number" or "#" is "four thousand four hundred
   seventy one" (after "order", "ticket", "case" only five digits or more are
   read one by one).
   Three-digit room and locker numbers are plain numbers ("room two hundred
   four"). "1:42" stays "one forty two" (minutes or hours is not known),
   "00:15" is "zero fifteen", "Genesis 1:1" is read as a ratio, "+1 (415)
   555-0132" does not say "plus", "$5-$10" has no "to". "9.30 to 11.30" is a
   time only after a word like "window", "open", "slot". No Roman numerals
   beyond "Henry VIII" and "World War II", nine currencies, 69 units.
   Continental figures are read only where the form leaves no doubt. "2.500"
   is "two point five zero zero" and "1.234.567" is read like a version;
   "12,50" with no sign or unit beside it is two numbers; three decimals
   after a comma ("1.234,567") are not joined. Thousands parted by spaces are
   one number only before a currency sign ("1 299,50 €") or when they end in
   "000"; "$10 000" and thousands parted by apostrophes ("1'299.50") are not
   handled. "4.300 €" is "four point three zero zero euros". A decimal comma
   is seen only next to the sign or the unit: in "1,5 to 2,5 kg" the first
   figure is still two numbers.
   Formulas are not understood. Dollar signs around one are dropped and the
   rest is read sign by sign ("$x^2$" is "X two"); a formula that opens with
   a figure is read as a price ("$3 + 5 = 8$" starts "three dollars").
   Comparisons: a name of several letters is not taken for a value, so
   "age > 18" says "age greater than eighteen" with no "is", and "count<n"
   with no space is not read at all. "*", "/" and "^" between names are
   silent ("a*b" is "A B"), arrows ("->", "=>") are not read, and "==" says
   "equals" twice.
7. **Capitals decide initialisms.** "WHO" is read "who"; an unknown word of up
   to five capitals is spelled; a two-letter one the dictionary knows as a
   word is read as that word ("ET" outside a time, "CT", "PO" outside a code).
   "A" is the article unless it is next to a digit, inside a code, ends a
   clause or follows "plan", "vitamin" and the like ("gate A 22" is still the
   article). Beside a sign or alone in brackets "a" and "A" are named
   ("a=10", "(a)"), but not in running words: "let A equal ten" and "k, a and
   n" still have the article. A lone "k" is "okay" only as a reply by itself;
   "k thanks" with no comma is the letter ("kay thanks"), and the "u" and "r"
   of chat stay "you" and "are" wherever no sign stands beside them.
   File names say "underscore" and "dash"; "/var" is spelled; an
   address ending that is a short common word ("about.me") is not recognised.
8. **Linking and flaps across words** follow the training tokens: only "it"
   and "at" flap before a vowel, although American speech flaps more.
9. **English only.** Letters with accents lose them; other scripts are
   skipped silently.
10. **Taught names are exact.** A name of several words is matched only as
    written, with nothing between its words ("Tung, tung, tung, sahur" falls
    back to the single words). There are no per-voice lists, and a
    pronunciation has to be written in CMUdict notation or in phones, not as
    "sah-HOOR".
11. **Python only.** The portable core is not written. The normalizer leans on
    regular-expression look-behind, which the port has to replace.
12. **Compounds and endings are conservative.** Two dictionary words of four
    letters or more ("sandpiper"); shorter halves go to the model.
13. **Sounds people write are a short list.** "Sh", "tsk", "psst", "pfft",
    "hmph", "psh", "hm" and "mm" are said as sounds, and so are "sss" and
    "zzz". "Ss" and "zz" are sounds only alone or among other sounds
    ("Sh! Ss! Tsk!"): next to a word they are letters, so "Ss, he said" is a
    hiss and "he said ss" is not. "Pst", "pff", "tch" and "ssh" are still
    named letter by letter ("PST" and "SSH" mean other things in capitals).
    "Ew" and "ahem" are left to the model, which gets them wrong. A sound
    with no vowel is one or two short phones; how well it is spoken depends
    on the voice.
14. **Braces are always a pronunciation.** Text in braces that is not one
    stops with an error instead of being read: "{name}", "{1, 2, 3}",
    "\frac{1}{2}". Of 1,985,590 real sentences 1,973 had such braces: 1,435
    of them in chat with an assistant (code, formulas), which is one chat
    sentence in 150.

## Building

A CPU is enough.

```bash
bash scripts/build.sh /work/dizione            # fetch, dictionary, model, homographs
G2P_ARGS="--device cuda" bash scripts/build.sh …  # the model in minutes on a GPU
```

`build.sh` fetches the two sources at pinned commits, estimates the alignment
(2 minutes), converts the dictionary (10 seconds), trains the unknown-word
model (about an hour and a half on 4 CPU cores, 15 minutes on a small GPU)
and the homograph classifiers (1 minute). After a change to a rule or a table only the
dictionary and the homograph step need to run again; the model predicts
CMUdict's notation, so it never has to be retrained for a rule change.

A rebuild from the sources gives the same dictionary, alignment table and
homograph weights byte for byte (checked on 14 Nov 2026 against the uploaded
files, and again after the second pass); the model's weights depend on the
hardware's arithmetic, so those are taken from the repository rather than
retrained. The hand-written tables in `data/` are read when the phonemizer
starts, so a new word, unit or abbreviation needs no rebuild.

The comparison with eSpeak on the tiny-voca corpus was made with
`scripts/compare_espeak.py` in the tiny-voca repository.

## Next steps

1. Fine-tune tiny-voca on these tokens.
2. Decide the language of the portable core. The Python here is the reference
   and `tricky_sentences.txt` plus the corpus tokens are its conformance test.
3. Pack the dictionary (3.4 MB of text now) and the weights (1.8 MB; 0.45 MB
   as 8-bit) for download size.
