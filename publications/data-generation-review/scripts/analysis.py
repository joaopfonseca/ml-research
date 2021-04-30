"""
Produce visualizations and tables to include in the paper.
"""
from os.path import join
import pickle
from gensim.models.ldamodel import LdaModel
from research.utils import (
    generate_paths
)


def main_venues(df, top_n=5):
    """
    Retrieve top 5 journals and conference proceedings with highest impact
    factor and published documents.
    """

    # Top n conferences - impact factor

    return


if __name__ == '__main__':
    data_path, results_path, analysis_path = generate_paths(__file__)

    results = pickle.load(open(join(results_path, 'results.pkl'), 'rb'))

    df = results['original_data']
    df['umap_x'], df['umap_y'] = results['document_umap_projections'].T

    top_journals, top_conferences = main_venues(df, top_n=5)
