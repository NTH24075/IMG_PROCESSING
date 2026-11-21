# Object Counting in Surveillance Images Using Segmentation and Contour Detection

## Abstract (v6, 16/11/2026) — review/tutorial style, structure of Uchida's abstract

> Citations use pandoc keys from ../../../references.bib. No numbers: the project's MAE results
> belong in the worked-example section, not in a review abstract.

This paper reviews image processing techniques for counting objects in surveillance images, with emphasis on segmentation and contour detection. Although it does not provide full technical details, it allows readers to grasp the main tasks of a counting pipeline and the typical tools for each task. Image segmentation is a large research area that partitions an image into regions corresponding to objects and background [@Pal_1993; @Minaee_2022]. As its main tasks, this paper introduces gray-level transformation, image filtering, thresholding, background subtraction, region-based segmentation and morphological processing [@Gonzalez_2018; @Otsu_1979; @Stauffer_1999; @Vincent_1991; @Haralick_1987]. Contour detection then delineates each segmented region, so that objects can be counted from the number, area and shape of these regions [@Rosenfeld_1966; @Canny_1986]. This paper also overviews learning-based alternatives, namely object detectors and density-based counters, which are trained on annotated images [@Dalal_2005; @Lempitsky_2010; @Zhang_2016; @Sindagi_2018]. Throughout the paper, it is emphasized that illumination change, cast shadows, noise and overlapping objects make surveillance images difficult targets for classical segmentation. A worked example on pedestrian images from fixed cameras illustrates how these steps are combined into a counting pipeline and where each step fails. This paper is expected to serve as a tutorial guide for readers who need to count objects without large annotated datasets.

(~200 words without citations.)
