# Object Counting in Surveillance Images Using Segmentation and Contour Detection

## Abstract (v5, 16/11/2026) — structure of the user-supplied tutorial-style model abstract

> Development-set numbers only. Citations use pandoc keys from ../../../references.bib.
> No bib entry supports shadow rejection yet.

This paper presents a training-free method for counting objects in fixed-camera surveillance images using segmentation and contour detection, with pedestrians as the target objects. Although it does not train any learned model, it shows how standard image processing operations can be combined into a complete and measurable counting system. Background subtraction detects moving objects by comparing each frame with a reference image of the empty scene [@Stauffer_1999; @Zivkovic_2004]. As its main processing steps, this paper introduces histogram matching, image filtering, threshold-based segmentation, morphological operations, shadow rejection and contour detection [@Gonzalez_2018; @Otsu_1979; @Haralick_1987; @Rosenfeld_1966]. Counting then converts the detected contours into a number of objects, a task usually addressed by learning from annotated images [@Lempitsky_2010; @Sindagi_2018]. This paper compares two training-free counting modules, a perspective-dependent area model and a pre-trained HOG–SVM person detector [@Dalal_2005]. On 76 development frames from two public video sequences, the area model reduced mean absolute error from 1.00 to 0.59 people per frame, whereas the detector reached 1.50. Throughout the paper, it is emphasized that illumination change, cast shadows and merged pedestrians remain difficult targets, and that all reported numbers come from development frames. This paper is expected to serve as a practical guide for applying classical image processing to object counting when annotated training data are unavailable.

(~200 words without citations.)
