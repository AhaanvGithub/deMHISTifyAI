# Literature reading: prompt check

The literature reading (backend/llm.py) explains the image panel's call from what was measured on the field and from
retrieved literature excerpts. It never sees the image and never overrules the panel. Each answer is checked in code
for rule breaks: an invented citation, overclaiming ("confirms", "diagnostic", "highly specific"...), evidence that
was not measured, a claim that something is absent ("lacking", "not seen"), a paper cited for a statement about this
particular field, and "SSA" instead of SSL.

Five test fields (borderline SSL on an unfamiliar slide, clear HP, clear SSL, mostly HP with one dilated crypt, no
crypts found), three runs each, Gemma-4 31B on Lumen:

| Prompt | Rule breaks per answer | Time per field |
|---|---|---|
| Rules only | 1.13 | 6.8 s |
| + "papers only for general facts; no absence claims" | 0.87 | 7.0 s |
| **+ a good and a bad example (in use)** | **0.80** | 7.1 s |

Example output (clear HP field): "Six measured crypts at the centre, middle left and middle right are round and
regular, which fits HP; regular crypts are typical of HP [5]." Some remaining counted breaks are the checker being
strict (a general citation in the same sentence as a location).

Design changes behind these numbers: the CONCH text-match "feature scores" are no longer given to the model (they were
guesses that contradicted each other and the model argued from them); the output is a summary of at most 45 words,
at most two measured points for the call and one against, and at most two checks that each name a place in the field;
three runs instead of five and five excerpts instead of eight (about 7-10 s instead of about 22 s). The confidence is
capped at 92%: on MHIST a single pathologist agrees with the 7-pathologist majority 83% of the time.
