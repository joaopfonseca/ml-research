======================================================================================================================================
Improving Imbalanced Land Cover Classification with K-means SMOTE: Detecting and Oversampling Distinctive Minority Spectral Signatures
======================================================================================================================================

[**Link to publication**](https://doi.org/10.3390/info12070266)

Abstract
========

Land cover maps are a critical tool to support informed policy development,
planning, and resource management decisions. With significant upsides, the
automatic production of Land Use/Land Cover maps has been a topic of interest
for the remote sensing community for several years, but it is still fraught
with technical challenges. One such challenge is the imbalanced nature of most
remotely sensed data. The asymmetric class distribution impacts negatively the
performance of classifiers and adds a new source of error to the production of
these maps. In this paper, we address the imbalanced learning problem, by
using K-means and the Synthetic Minority Oversampling TEchnique (SMOTE) as an
improved oversampling algorithm.  K-Means SMOTE improves the quality of newly
created artificial data by addressing both the between-class imbalance, as
traditional oversamplers do, but also the within-class imbalance, avoiding the
generation of noisy data while effectively overcoming data imbalance.  The
performance of K-means SMOTE is compared to three popular oversampling methods
(Random Oversampling, SMOTE and Borderline-SMOTE) using seven remote sensing
benchmark datasets, three classifiers (Logistic Regression, K-Nearest
Neighbors and Random Forest Classifier) and three evaluation metrics using a
5-fold cross-validation approach with 3 different initialization seeds. The
statistical analysis of the results show that the proposed method consistently
outperforms the remaining oversamplers producing higher quality land cover
classifications. These results suggest that LULC data can benefit
significantly from the use of more sophisticated oversamplers as spectral
signatures for the same class can vary according to geographical distribution.
