# Abstract (v3, 16/11/2026) — 5-part structure

> Development-set numbers only. Replace the `[TEST: ...]` sentence after the `freeze-test` run.

<!-- 1. Context (present tense) -->
Counting people in fixed-camera surveillance images supports crowd monitoring and the management of public spaces. Most current counters learn from annotated training images, which are costly to collect for every new camera. Background subtraction needs no training, but it miscounts when lighting changes, shadows join the foreground or nearby pedestrians merge into one blob.
<!-- 2. Objective -->
This study aimed to make a training-free, background-subtraction counter robust to these three failures while keeping millisecond-scale processing.
<!-- 3. Methods -->
The pipeline matched each frame's histogram to the reference background and rejected shadow pixels using an intensity ratio and a gradient difference. A perspective-dependent area model then converted each blob's area into a person count. All parameters were fitted on 76 doubly reviewed development frames from two public scenes, with a pre-trained HOG–SVM detector as reference.
<!-- 4. Results -->
Mean absolute error (MAE) fell from 1.00 to 0.59 people per frame against a one-blob-one-person baseline, compared with 1.50 for the detector. Ablation attributed most of this gain to the area model. [TEST: On 75 held-out frames from an unseen scene, MAE was X against Y for the baseline.]
<!-- 5. Conclusion (present tense) -->
Photometric normalisation and a calibrated area model therefore make classical background subtraction a fast, training-free counting option for static cameras with a clean background.

(~190 words without the TEST sentence.)
