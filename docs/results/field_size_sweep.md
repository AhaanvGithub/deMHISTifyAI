# Field size: does a bigger field help on other labs' slides?

The scale audit (scale_audit.md) found MHIST tiles likely cover about 720-860 um, not the nominal 448 um. If the app's
448 um field shows the models a more zoomed-in view than they learned from, a bigger field might recover SSL on other
labs' slides. research/field_size_panel_sweep.py scores outside fields with the app's own panel at several sizes
(partial run: 448, 650 and 790 um).

| Panel result by field size | 448 um (app) | 650 um | 790 um |
|---|---|---|---|
| HunCRC slide 135 (SSL slide): fields called SSL | 4% | 2% | 4% |
| HunCRC HP slide 018: fields called HP | 93% | not run | 98% |
| UniToPatho HP: called HP | 99.6% | not run | 99.4% |
| UniToPatho normal colon: called HP | 100% | not run | 100% |
| Outside balanced accuracy | 51% | not run | ~51% |
| MHIST test (its own tiles), for reference | 88.8% | | |

A bigger field keeps HP correct but does not bring out SSL, so field size is not why outside SSL is missed; the app
keeps 448 um. Caveats: slide 135's lesion area is not marked (much of it is normal or unusable tissue, so a low SSL
share is partly correct); 120 fields per slide per size; SPIDER and 550/720/860 um were not run.
