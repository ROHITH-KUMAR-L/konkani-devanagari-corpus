# Konkani (Devanagari) NER annotation guidelines

Version 0.1 (draft, 21 Nov 2026). Applies to `data/eval/ner/*.conll`. Revise this file whenever the team settles a disputed case, and add the case to section 7.

## 1. Tag set and scheme

We use the five types from PRD FR-P3-2, encoded as **BIO** (`B-X` starts an entity, `I-X` continues it, `O` is outside).

| Type | What it covers | Examples |
|---|---|---|
| **PER** | Named people, including deities, saints and mythological or fictional characters addressed by name | प्रमोद सावंत, बाकीबाब, कृष्ण, जेजू, पांडवांनी |
| **LOC** | Named places: countries, states, cities, villages, wards, rivers, seas, mountains, regions, named roads and named buildings or monuments | गोंय, पणजी, कळंगूट, मांडवी, कोंकण, कफ परेड, फातिमा चर्च |
| **ORG** | Named organisations: government bodies and departments, parties, companies, colleges and schools, associations, newspapers, sports clubs, dynasties | गोंय विद्यापीठ, कोंकणी अकादमी, भाजप, पीडब्ल्यूडी, शिक्षण संचालनालय |
| **MISC** | Other proper names: languages, nationalities and demonyms, religions and their adjectives, festivals and named events, titles of books, films, songs and plays, awards, named schemes, laws and frameworks, product names | कोंकणी, गोंयकार, हिंदू, चवथ, नोबॅल पुरस्कार, ऑस्कर |
| **DATE** | Absolute or named calendar expressions: day + month (+ year), month + year, years, centuries, named weekdays | 17 डिसेंबर 2026, 1912, १९ व्या शतमानाचे, सोमारा |

Not entities: common nouns and job titles (मुख्यमंत्री, सरकार, पुलीस used generically), times of day (सकाळीं 10.00 वरांचेर), durations (दोन दीस), relative dates (फाल्यां, काल), numbers, money and percentages.

## 2. Tokens

Tokens are those of `scripts/phase3/sample_ner_sentences.py`, which uses the corpus tokeniser (`scripts/corpus_utils.py`, `TOKEN_RE`): punctuation is dropped, and numbers keep their digits so that DATE can be marked. **Do not split or merge tokens while annotating.** If the tokenisation is wrong, write the problem in your notes and tag the tokens as they are.

## 3. Konkani-specific rules

### 3.1 Case suffixes and postpositions written on the name
Konkani attaches case markers to the noun: गोंयांत (in Goa), भारताचो (of India), मुंबयच्या, पुलिसांक, सावंतान, फोंडेंचो. **The whole token is tagged.** We do not split the suffix off, so the entity span is the full token.

- `गोंयांत` → `B-LOC`
- `आसोलकारांच्या` in `नाना आसोलकारांच्या` → `B-PER I-PER`
- Free-standing postpositions that are separate tokens (`हांचे`, `खातीर`, `वरवीं`, `कडेन`) are **outside** the entity: `जयंती नायक हांचे` → `B-PER I-PER O`.
- Emphatic or additive clitics (`-च`, `-य`, `-ूय`) are part of the token and get the entity tag: `रोहनाकय` → `B-PER`.

### 3.2 Honorifics and titles
Titles before a name are **not** part of PER: `डॉ`, `प्रा`, `श्री`, `श्रीमती`, `सौ`, `ॲड`, `स्वामी`, `पंडित`, `ह भ प`, `संत`, `देव`, `सायब`, and role words such as `मुख्यमंत्री`, `आमदार`, `अध्यक्ष`.

- `डॉ प्रमोद सावंत` → `O B-PER I-PER`
- `स्वामी दयानंद सरस्वती` → `O B-PER I-PER`
- `देव रवळनाथ` → `O B-PER`

Exception: honorifics that have become part of the name (`बाकीबाब`, `शणै गोंयबाब`, `भाऊसायब`, `-बाय` in `शांताबाय`) stay inside the token and the token is PER.

### 3.3 Names with initials and transliterated English names
Initials written as separate Devanagari tokens belong to the name: `बा भ बोरकर` → `B-PER I-PER I-PER`; `पुंडलिक डी गायतोंडे` → three PER tokens. Transliterated English and Portuguese names (`जिओविनो फर्नांडीस`, `सेल्सो फर्नांडीस`, `हेन्ऱी टॉमस कोलब्रूक`) follow the same rules as Konkani names. Abbreviations spelled out letter by letter (`एल आय सी`) are tagged as one span. Latin-script tokens, if any remain, are tagged like any other token.

### 3.4 Goa place names
- The state is `गोंय` (also `गोंयां`). Talukas (बार्देश, तिसवाडी, सासश्टी, सत्तरी, काणकोण ...), towns and villages (पणजी, म्हापशें, मडगांव, फोंडें, कळंगूट ...), and wards (`-वाडो`, `-भाट`) are LOC.
- Spelling variants of the same place are all LOC: म्हापशें / म्हापसा / म्हापश्यां; फोंडें / फोंडे / फोंडा.
- `गोंयकार` (Goan person/people) is a demonym, so it is **MISC**, not LOC.
- A place name used as part of an organisation name belongs to the ORG span: `गोंय विद्यापीठ` → ORG, `कळंगूट पंचायतीन` → ORG.
- Datelines at the start of news items (`पणजी गोंय शुक्रार 20 नोव्हेंबर 2026`) are tagged normally: LOC LOC DATE.

### 3.5 Metonymy
Annotate the **literal type**, not the intended referent. A country playing cricket (`भारतान 209 धांवड्यो काडल्यो`) stays LOC. A government is ORG only when the text names it as a government (`भारत सरकारान` → ORG), and the bare place stays LOC.

### 3.6 Adjacent and nested entities
- Two adjacent names of the same type start separate spans: `भारत रशिया` → `B-LOC B-LOC`.
- No nesting: annotate the **longest** (outermost) span. `कोंकणी भाशा मंडळ` is ORG, and the language inside it is not tagged separately. `राष्ट्रीय अभ्यासक्रम आराखडो 2023` is one MISC span, and the year inside it is not a separate DATE.
- Coordination: `उराल न्हंय आनी उराल पर्वत` is two LOC spans, and `आनी` is O.

### 3.7 Generic words for features
`न्हंय`, `दर्या`, `पर्वत`, `म्हासागर` and `खंड` go inside the LOC span when they are part of the proper name (`कॅस्पियन दर्या`, `युरोप खंड`). They are O when used generically.

### 3.8 OCR noise and mixed language
- If an OCR error merges a name with the next word (`फोंडेंच्याभाजप`), tag the token with the type of the entity that **starts** it.
- If the text is too garbled to identify the name, tag O and record the sentence ID in your notes.
- Hindi and Marathi sentences that slipped into the corpus are annotated by the same rules.

## 4. Decision checklist (per candidate span)
1. Is it a proper name, or a DATE expression of the kind listed in section 1? If not, tag it O.
2. Leave out leading titles and honorifics (section 3.2).
3. Keep attached suffixes, and leave out separate postposition tokens (section 3.1).
4. Pick the literal type (section 3.5). If still unsure between ORG and MISC, choose ORG for things that act (employ people, make decisions) and MISC for everything else.
5. Take the longest span (section 3.6).

## 5. Workflow
1. `scripts/phase3/sample_ner_sentences.py` writes the stratified pool `data/eval/ner/pool.conll` (3,000 sentences, seed 2026). Any prefix of the pool is itself stratified.
2. The pool has been split among the four members. The first 300 sentences carry an **AI first-pass silver annotation** (`silver_spans.txt`, built into `silver_*.conll` by `build_ner_silver.py`). Correct these first.
3. Annotate in Doccano or Label Studio (import CoNLL, export CoNLL), or edit `silver_spans.txt` directly and rerun the build script.
4. A shared subset of 200 or more sentences is annotated by all four members independently, and Cohen's κ is computed on token tags (PRD FR-P3-3, target κ ≥ 0.7).
5. Disagreements are settled in a meeting, and each new rule is added below.

## 6. Status of the silver data
The 300 silver sentences were labelled by an AI assistant (Claude) as a first pass. They are **not gold**. Known weak spots that need human review:
- Deities and epic characters were tagged PER. Confirm this is the team's choice.
- Dynasties (`सातवाहनाच्या`, `मौर्य`) were tagged ORG.
- Titles of songs, films and articles (MISC) were guessed from context.
- Garbled OCR sentences (e.g. ner00027, ner00061, ner00163, ner00245) contain names that may be misread.
- Some ambiguous tokens were left O (e.g. `कोयसांवाक` in ner00226, `सिगा` in ner00096).

## 7. Resolved cases log
| Date | Case | Decision |
|---|---|---|
| 2026-11-21 | Language names used as adjectives (`कोंकणी नाटक`) | MISC on the language token only |
| 2026-11-21 | Country as sports team | LOC (literal type) |
