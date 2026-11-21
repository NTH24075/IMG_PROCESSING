# Abstract (v4, 16/11/2026) — 5-part structure, cited

> Development-set numbers only. Replace the `[TEST: ...]` sentence after the `freeze-test` run.
> Citations use pandoc keys from ../../../references.bib. No entry in the bib supports shadow detection.

<!-- 1. Context (present tense) -->
Automatic people counting in fixed-camera video supports crowd monitoring [@Chan_2008]. Most current counters, from density-map regression to class-agnostic counting, are trained on annotated images [@Lempitsky_2010; @Zhang_2016; @Sindagi_2018; @Ranjan_2021]. Background subtraction requires no labelled data [@Stauffer_1999; @Zivkovic_2004], but its counts are sensitive to illumination change, cast shadows and pedestrians whose silhouettes merge into one blob.
<!-- 2. Objective -->
This study aimed to reduce these three error sources in a training-free background-subtraction counter while keeping processing at the millisecond scale.
<!-- 3. Methods -->
Each frame was matched to the reference background by histogram specification [@Gonzalez_2018], and shadow pixels were rejected using an intensity ratio and a gradient difference. A perspective-dependent area model then converted the area of each blob into a person count. Parameters were selected on 76 development frames from two public video sequences, with counts verified by two reviewers.
<!-- 4. Results -->
Relative to a one-blob-one-person baseline, mean absolute error (MAE) decreased from 1.00 to 0.59 people per frame. A pre-trained HOG–SVM detector [@Dalal_2005] reached 1.50 on the same frames (paired difference 0.91, 95% CI 0.33–1.45). Removing the area model restored the baseline error, whereas the other components changed MAE only within sampling error. [TEST: On 75 held-out frames from an unseen scene, MAE was X against Y for the baseline.]
<!-- 5. Conclusion (present tense) -->
These results suggest that a calibrated area model, supported by photometric normalisation, makes background subtraction a fast, training-free counting option for static cameras with a clean background.

(~200 words without citations and the TEST sentence.)
