# dizione

dizione converts English text to phonemes. It reads numbers, dates, money,
units, abbreviations and chat text the way a person would say them, picks the
right pronunciation for words like "read" and "live" from context, and guesses
words it has never seen with a small model.

It was written as the front end of the
[tiny-voca](https://huggingface.co/n4ze3m/tiny-voca) text-to-speech model, and
has no dependency other than NumPy.

| | |
|---|---|
| Language | American English |
| Output | 59 phones, stress as a separate 0/1/2 feature, word and punctuation tokens |
| Dictionary | 125,000 words, converted from CMUdict by rule |
| Unknown words | 459k-parameter model, NumPy inference |
| Homographs | 162 words, chosen from context |
| Speed | about 4,400 sentences per second on one CPU core |
| Data size | 5.4 MB |
| License | Apache-2.0 |

## Usage

```bash
pip install -e .
export DIZIONE_DATA=/path/to/built     # see "Data" below
python -m dizione --words "Dr. Patel paid $12.99 on 3/14."
```

```python
import dizione

result = dizione.phonemize("I read the book you read yesterday.")
print(result)            # <bos> aɪ <sp> ɹ ˈiː d <sp> ð ə <sp> b ˈʊ k ...
result.symbols           # ("<bos>", "aɪ", "<sp>", "ɹ", "iː", "d", ...)
result.ids               # integer ids into dizione.symbols.SYMBOLS
result.stress            # 0 none, 1 primary, 2 secondary, one per symbol
```

Your own pronunciations take priority over everything else. Give them in
CMUdict notation or directly as phones:

```python
from dizione.engine import Phonemizer

p = Phonemizer(overrides={"kubernetes": "k ˌuː b ɚ n ˈɛ ɾ iː z", "Xochitl": "S OW1 CH IY0 L"})
```

A file works too: set `DIZIONE_OVERRIDES` to a text file with one word per
line, two spaces, then the pronunciation. For a single word in one text, write
it inline: `Say {HH AH0 L OW1} to everyone.`

## How it works

A sentence goes through these steps in order:

1. Text rules turn numbers, ordinals, years, money, dates, times, phone
   numbers, units, abbreviations, symbols, web addresses and file names into
   words. Emoticons and emoji are dropped. Chat forms are expanded ("btw") or
   spelled ("lol").
2. Your overrides and inline pronunciations are applied.
3. Each remaining word is looked up: a hand-written exception list, then the
   dictionary, then word-building rules (plurals, "-ing", "-ed", compounds),
   then the unknown-word model.
4. Homographs are resolved from the surrounding words.
5. Sentence rules apply what depends on neighbours: weak forms of function
   words, flaps, linking.

[docs/design.md](docs/design.md) describes each step, the measurements and the
known errors.

## Evaluation

| Test | Result |
|---|---|
| Unknown-word model, 6,235 held-out CMUdict words | 7.9% phoneme error, 32.4% word error |
| Same, ignoring stress | 6.1% phoneme error, 26.5% word error |
| Homographs, held-out sentences | 95.4% correct (always choosing the most common reading: 84.0%) |
| Test sentences in `tests/data/tricky_sentences.txt` | all pass |

## Data

The dictionary, the unknown-word model and the homograph weights are built
from two sources and are not stored in this repository. A built copy is in the
`phonemizer/` folder of https://huggingface.co/n4ze3m/tiny-voca:

```bash
hf download n4ze3m/tiny-voca --include "phonemizer/*" --local-dir tiny-voca
export DIZIONE_DATA=tiny-voca/phonemizer
```

To build it yourself:

```bash
pip install -e ".[build]"
bash scripts/build.sh /path/to/work      # fetches the sources, writes /path/to/work/built
export DIZIONE_DATA=/path/to/work/built
```

The build takes about an hour and a half on four CPU cores, or 15 minutes with
`G2P_ARGS="--device cuda"`.

## Limitations

- American English only.
- About one unknown word in three has an error, usually one vowel or the
  stress. Foreign names are the worst case.
- Only 162 homographs are resolved from context.
- Slash dates are read month first unless the text makes day first clear.
- Whether a bare number is an amount or a reference ("ticket 4471") is a guess.
- Python only. There is no C, Rust or JavaScript port yet.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

Tests that need the built data are skipped when `DIZIONE_DATA` is not set.

## License

Code: Apache-2.0.

The built data is derived from CMUdict (Carnegie Mellon University, BSD-style
license) and the Wikipedia Homograph Data (Google, Apache-2.0). Their license
files are copied into the built directory.

## Citation

```bibtex
@software{nazeem2026dizione,
  author = {Nazeem, Muhammed},
  title  = {dizione: English text to phonemes},
  year   = {2026}
}
```

Author: Muhammed Nazeem ([ORCID 0009-0002-9991-0217](https://orcid.org/0009-0002-9991-0217)).
