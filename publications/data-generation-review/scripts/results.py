"""
Generate the main experimental results.
"""
from os.path import join
from itertools import combinations
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from gensim.models import Phrases
from gensim.parsing.preprocessing import preprocess_documents
from gensim.sklearn_api.d2vmodel import D2VTransformer
from umap import UMAP
from research.utils import (
    load_datasets,
    generate_paths
)
import sompy

preprocess = FunctionTransformer(preprocess_documents)

# Set up document embedding pipeline
encoder = Pipeline([
    ('preprocess', preprocess),
    ('doc2vec', D2VTransformer(
        size=25,
        iter=100,
        min_count=10,
        seed=42,
        workers=8
    )),
    ('stdscaler', StandardScaler()),
])


def make_som(X):
    """Fit a SOM and export the SOM's grid info."""
    # Fit SOM
    np.random.seed(42)
    som = sompy.SOMFactory.build(
        X, (30, 30),
        mapshape='planar', lattice='hexa',
        normalization='var', initialization='pca',
        neighborhood='gaussian', training='batch',
    )
    som.train(
        train_rough_len=50,
        train_finetune_len=100,
        n_job=8, verbose='info'
    )

    # Set up U-Matrix
    u = sompy.umatrix.UMatrixView(
        9, 9, 'U-Matrix', show_axis=True, text_size=8, show_text=True
    )

    umatrix = u.build_u_matrix(som)

    # som units values in the input and grid spaces
    codebook = som.codebook.matrix
    bmus = pd.DataFrame(codebook)
    indices = np.indices(umatrix.shape)
    bmus['row'] = indices[0].flatten()
    bmus['col'] = indices[1].flatten()
    bmus['umat_val'] = umatrix.flatten()

    X_bmus = som.find_bmu(X)[0]

    return X_bmus, bmus


def _document_network_connections(edges, weight):
    """Helper function to convert list of edges to dataframe."""
    edges = pd.DataFrame(edges)
    edges['weight'] = weight
    return edges


def lda_preprocessing(documents):
    """Run an LDA analysis."""
    preprocessed_docs = preprocess_documents(documents)
    bigram = Phrases(preprocessed_docs, min_count=5, threshold=10)
    trigram = Phrases(bigram[preprocessed_docs], threshold=10)

    docs = [bigram[doc] for doc in preprocessed_docs]
    docs = [trigram[doc] for doc in docs]

    return docs


if __name__ == '__main__':

    data_path, results_path, _ = generate_paths(__file__)

    df, df_journals = [
        data[1]
        for data in load_datasets(
            data_path, suffix='.db', target_exists=False
        )
    ]

    # Embed documents based on Abstract
    X = encoder.fit_transform(df.Abstract)

    # Get umap projections
    projections = UMAP(random_state=42).fit_transform(X)

    # assign a unit ID to each document
    df['docs_bmu'], bmus = make_som(X)

    bmus = bmus.join(
        df.groupby('docs_bmu').size().rename('docs_count')
    )

    # Save encoder
    pickle.dump(
        encoder, open(join(results_path, 'encoder.pkl'), 'wb')
    )

    # Create weighted keyword network data
    df_net = df[['Author Keywords', 'Cited by']].copy()
    df_net['Author Keywords'] = df_net['Author Keywords'].apply(
        lambda x: list(combinations(x.split('; '), 2)) if x else None
    )
    df_net = pd.concat(
        df_net.apply(
            lambda row: _document_network_connections(*row), axis=1
        ).tolist()
    )
    df_net.rename(columns={0: 'source', 1: 'target'}, inplace=True)
    df_net.source = df_net.source.str.lower()
    df_net.target = df_net.target.str.lower()
    df_net = df_net.groupby(['source', 'target'])\
        .agg({np.size, np.mean})['weight'].reset_index()\
        .rename(columns={'mean': 'Avg cites', 'size': 'Nbr of documents'})
    df_net['weight'] = np.log(df_net['Avg cites'] * df_net['Nbr of documents'])

    df_net = df_net[df_net['Nbr of documents'] > 1]

    # Preprocess documents for LDA
    lda_docs = lda_preprocessing(df.Abstract)

    # Save results
    output = {
        'original_data': df,
        'document_umap_projections': projections,
        'document_embeddings': X,
        'som_grid': bmus,
        'network_data': df_net,
        'preprocessed_docs_lda': lda_docs,
        'journal_analysis_data': df_journals
    }
    pickle.dump(
        output, open(join(results_path, 'results.pkl'), 'wb')
    )
