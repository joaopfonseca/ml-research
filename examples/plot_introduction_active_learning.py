"""
Introduction to Active Learning
===============================

This example explains what Active Learning (AL) is and how ml-research can be used to run
AL simulations.

We will focus exclusively on the ``StandardAL`` object, which is the typical/classical
process used in AL.

Let's start by setting up our environment.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlresearch.active_learning import StandardAL

# This will apply a nice formatting style to our plots
from mlresearch.utils import set_matplotlib_style, feature_to_color

set_matplotlib_style(font_size=16, use_latex=False)


# Set up some environment variables
RNG_SEED = 42
labels = np.array([-1, 0, 1])
color_labels = {
    label: color
    for label, color in zip(labels, feature_to_color(labels + 1, cmap="Accent"))
}


# Define some helper functions
def plot_data(X, y, classifier=None, ax=None):

    if ax is None:
        fig, ax_ = plt.subplots()
    else:
        ax_ = ax

    if classifier is not None:
        DecisionBoundaryDisplay.from_estimator(classifier, X=X, alpha=0.2, ax=ax_)

    for label in labels:
        mask = y == label
        ax_.scatter(
            X[mask, 0], X[mask, 1], c=color_labels[label], label=label, alpha=0.5
        )

    if ax is None:
        plt.legend()
        plt.show()
    else:
        return ax_


######################################################################################
# We can now generate a simple mock dataset with 2 features and 2 target classes.
# Our goal is to produce a high-performing classifier that will be able to distinguish
# the 2 classes:

X, y = make_classification(
    n_samples=500, n_features=2, n_informative=2, n_redundant=0, random_state=RNG_SEED
)
plot_data(X, y)

######################################################################################
# What is Active Learning?
# ------------------------
#
# Now that we have our problem set up, we can discuss what AL actually is.
# AL is most commonly used when we have a large pool of unlabeled data. In such cases,
# if we want to train a classifier, we will need to label/annotate some of these data
# points in order to produce a training dataset. However, randomly selecting data points
# to form this training dataset is very inefficient; the annotation process can be time
# consuming or expensive (and sometimes both!). AL allows this process to be much more
# efficient, since it attempts to retrieve the most informative data points to the
# learning process.
#
# **The goal of AL is to find the smallest possible data subset that will allow a
# classifier to achieve the best possible performance.**
#
# Let's apply this description to our dataset by assuming it is unlabeled at an initial
# state:

y_known = np.zeros(y.shape) - 1

plot_data(X, y_known)

######################################################################################
# Here, the label `-1` means the label of a given point is unknown. Since there is no
# information this stage, we currently have no way to select points for annotation in an
# informed way.
#
# Let's begin by selecting two random points (one from each class) and use those to train
# a classifier. To do this, we can use the `StandardAL` class.

classifier = make_pipeline(
    MinMaxScaler(), MLPClassifier((20, 20), max_iter=3000, random_state=RNG_SEED)
)

al = StandardAL(
    classifier=classifier,
    acquisition_func="breaking_ties",
    n_init=2,
    budget=4,
    random_state=RNG_SEED,
)
al.initialization(X, y)
y_known[al.labeled_pool_] = y[al.labeled_pool_]

# At this point, we only collected 2 labeled points, no classifier has been trained yet
plot_data(X, y_known)

######################################################################################
# The acquisition function is defined to ensure the following annotation stages are done
# based on `breaking ties`, which will quantify the uncertainty of the classifier when
# predicting the labels for the unlabeled points. The unlabeled points with the highest
# uncertainty are the ones we expect to be the most valuable to annotate and include in
# the training dataset. This is the core concept of AL.
#
# Usually, the points selected for annotation are the ones closest to the decision
# boundary. This is because the classifier is most uncertain about these points, and
# they are therefore the most informative.
#
# Let's select 4 additional points for annotation:

al.iteration(X, y)
y_known[al.labeled_pool_] = y[al.labeled_pool_]
plot_data(X, y_known, al.classifier_)

######################################################################################
# Repeat this process until the model reaches a satisfactory performance level. Let's
# see how the classifier evolves as we annotate more points:

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for i in range(18):
    al.iteration(X, y)
    y_known[al.labeled_pool_] = y[al.labeled_pool_]
    if i % 2 == 0:
        ax = axes.flatten()[i // 2]
        plot_data(X, y_known, al.classifier_, ax=ax)

plt.show()

######################################################################################
# As we can see, the classifier is able to learn the decision boundary more accurately
# as we annotate more points. At this point, our model would classify the remaining
# points as follows:

y_pred = al.classifier_.predict(X)
print("F1 Score:", f1_score(y, y_pred, average="weighted"))
plot_data(X, y_pred, al.classifier_)

######################################################################################
# We can simplify the process of running AL experiments by using the
# `fit_predict` method, which will run the entire process for us. We will run an
# experiment with more points being labeled per iteration, until all points are labeled,
# while keeping track of the test score at each iteration:

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RNG_SEED
)

classifier = make_pipeline(
    MinMaxScaler(), MLPClassifier((20, 20), max_iter=3000, random_state=RNG_SEED)
)

al = StandardAL(
    classifier=classifier,
    acquisition_func="breaking_ties",
    n_init=10,
    budget=10,
    evaluation_metric="f1_weighted",
    random_state=RNG_SEED,
)
al.fit(X, y, X_test=X_test, y_test=y_test)

test_scores = [al.metadata_[i]["test_score"] for i in range(1, al.max_iter_)]
plt.plot(test_scores)
plt.show()

######################################################################################
# As we can see, the performance of the classifier improves as we annotate more points,
# up to a certain point. After that, the performance stabilizes or even decreases, since
# the classifier is overfitting to the training data.
#
# We can visualize the decision boundaries of the classifier at different
# iterations to show this effect:

fig, axes = plt.subplots(1, 4, figsize=(15, 5))
for iter_, ax in zip([1, 12, 24, al.max_iter_], axes.flatten()):
    metadata = al.metadata_[iter_]
    y_known = np.zeros(y.shape) - 1
    y_known[metadata["labeled_pool"]] = y[metadata["labeled_pool"]]
    plot_data(X, y_known, metadata["classifier"], ax=ax)
    ax.set_title(f"Iteration {iter_}")

ax.legend()
plt.show()
