=====================================================================================================================================
Increasing the Effectiveness of Active Learning: Introducing Artificial Data Generation in Active Learning for Land Use/Land Cover Classification
=====================================================================================================================================

Abstract
========

In remote sensing, Active Learning (AL) has become an important technique
to collect informative ground truth data "on-demand" for supervised
classification tasks.  In spite of its effectiveness, it is still
significantly reliant on user interaction, which makes it both expensive
and time consuming to implement.  Most of the current literature focuses on
the optimization of AL by modifying the selection criteria, the chooser
and/or predictors used.  Although improvements in these areas will result
in more effective data collection, the use of artificial data sources to
reduce human-computer interaction remains unexplored. In this paper, we
introduce a new component to the typical AL framework, the data generator,
a source of artificial data to reduce the amount of user-labeled data
required in AL\@.  The implementation of the proposed AL framework is done
using SMOTE and Geometric SMOTE as data generators.  We compare the new AL
framework to the original one using similar acquisition functions and
predictors over three AL-specific performance metrics in seven benchmark
datasets. We show that this modification to the AL framework significantly
reduces cost and time requirements for a successful AL implementation in
the context of remote sensing. 
