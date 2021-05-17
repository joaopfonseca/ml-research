"""
Produce visualizations and tables to include in the paper.
"""
from os.path import join
import pickle
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel, CoherenceModel
import networkx as nx
import networkx.algorithms.community as nxcom
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from rich.progress import track
from rlearn.utils import check_random_states
from research.utils import (
    generate_paths,
    load_plt_sns_configs
)

RANDOM_STATE = 42
LDA_TOPICS = 8


def main_venues(df, top_n=10, min_docs=10):
    """
    Retrieve top n journals and conference proceedings with highest
    citations/document ratio.
    """
    df_journals = df[~df['is_conference'].astype(bool)]
    df_conference = df[df['is_conference'].astype(bool)]

    # Top n venues - citations/document ratio
    analyses = []
    for df_ in [df_journals, df_conference]:
        df_ = df_\
            .groupby('Source title')\
            .agg({'Cited by': [np.size, np.sum]})['Cited by']

        df_['Average'] = (
            df_['sum'] / df_['size']
        )
        df_.sort_values(
            'Average',
            ascending=False,
            inplace=True
        )

        df_ = df_[df_['size'] >= min_docs].head(top_n)
        df_.rename(
            columns={'size': 'Publications', 'sum': 'Citations'},
            inplace=True
        )

        # Format values
        df_['Average'] = df_['Average'].apply(lambda x: '{0:.2f}'.format(x))
        df_['Citations'] = df_['Citations'].astype(int)
        df_['Publications'] = df_['Publications'].astype(int)

        analyses.append(df_)

    return analyses


def per_year_publications(df):
    """Plot the number of documents per year"""

    df['Is cited'] = df['Cited by'].isna().apply(
        lambda x: 'Cited' if x else 'Uncited'
    )

    df_ = df.groupby(['Year', 'Is cited'])\
        .size().to_frame('count').reset_index()\
        .pivot('Year', 'Is cited', 'count').fillna(0)

    ax = plt.subplot(111)
    df_[['Uncited', 'Cited']].plot.area(
        ax=ax,
        figsize=(7.5, 3),
        color={
            'Cited': 'steelblue',
            'Uncited': 'indianred'
        },
        alpha=.7
    )
    plt.xlabel('Year', fontsize=8)
    plt.ylabel('Number of Documents', fontsize=8)
    ax.set_xlim(2006, 2021)

    plt.savefig(
        join(analysis_path, 'area_chart_cited_documents.pdf'),
        bbox_inches='tight'
    )
    plt.close()


def compute_coherence_values(
    dictionary, corpus, texts, limit, start=2, step=1, n_inits=5,
    random_state=None
):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with
    respective number of topics
    """
    # Setup random states
    random_states = check_random_states(random_state, n_inits)

    # Compute coherence for each value of num_topics
    coherence_values_temp = []
    coherence_values = []
    for num_topics in track(
        range(start, limit, step),
        description='Mean coherence scores'
    ):
        for rs in random_states:
            model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=num_topics,
                passes=20,
                alpha=.1,
                eta='auto',
                random_state=rs
            )
            coherencemodel = CoherenceModel(
                model=model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )
            coherence_values_temp.append(coherencemodel.get_coherence())
        coherence_values.append(np.mean(coherence_values_temp))

    return coherence_values


def get_dominant_topic(lda_model, corpus):
    """
    Extract dominant topic for each document and its percentage contribution.
    """
    doc_topics = []
    doc_topics_percent = []
    for document in corpus:
        topics_dist = lda_model.get_document_topics(document)
        dom_topic, percent = max(topics_dist, key=lambda item: item[1])
        doc_topics.append(dom_topic+1)
        doc_topics_percent.append(percent)
    return doc_topics, doc_topics_percent


def get_secondary_topic(lda_model, corpus):
    """
    Extract dominant topic for each document and its percentage contribution.
    """
    doc_topics = []
    doc_topics_percent = []
    for document in corpus:
        topics_dist = lda_model.get_document_topics(document)
        if len(topics_dist) > 1:
            topics_dist.remove(max(topics_dist, key=lambda item: item[1]))
        dom_topic, percent = max(topics_dist, key=lambda item: item[1])
        doc_topics.append(dom_topic+1)
        doc_topics_percent.append(percent)
    return doc_topics, doc_topics_percent


def get_terciary_topic(lda_model, corpus):
    """
    Extract dominant topic for each document and its percentage contribution.
    """
    doc_topics = []
    doc_topics_percent = []
    for document in corpus:
        topics_dist = lda_model.get_document_topics(document)
        if len(topics_dist) > 1:
            topics_dist.remove(max(topics_dist, key=lambda item: item[1]))
        if len(topics_dist) > 1:
            topics_dist.remove(max(topics_dist, key=lambda item: item[1]))
        dom_topic, percent = max(topics_dist, key=lambda item: item[1])
        doc_topics.append(dom_topic+1)
        doc_topics_percent.append(percent)
    return doc_topics, doc_topics_percent


def lda_analysis(corpus, num_topics=30, line_plot=True):
    """Perform an LDA analysis."""

    # Create Dictionary
    corpus_dict = Dictionary(corpus)

    # Filtering extremes by removing tokens occuring in less than 5 documents
    # and have occured in more than 90% documents.
    corpus_dict.filter_extremes(no_below=5, no_above=.45)

    # Create Corpus: Term Document Frequency
    corpus_ = [corpus_dict.doc2bow(doc) for doc in corpus]

    # Adding the TF-IDF for better insight
    tfidf = TfidfModel(corpus_)
    tfidf_corpus = tfidf[corpus_]

    # Plot coherence results
    if line_plot:
        coherence_values = compute_coherence_values(
            corpus=tfidf_corpus,
            dictionary=corpus_dict,
            texts=corpus,
            start=2,
            limit=60,
            n_inits=5,
            random_state=RANDOM_STATE
        )

        x = range(2, 60)
        plt.plot(x, coherence_values)
        plt.axvline(x=num_topics, linestyle='--', color='red')
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.savefig(join(analysis_path, 'lda_coherence_analysis.pdf'))
        plt.close()

    # Fit LDA
    lda = LdaModel(
        corpus=tfidf_corpus,
        id2word=corpus_dict,
        num_topics=num_topics,
        passes=20,
        alpha=.1,
        eta='auto',
        random_state=RANDOM_STATE
    )

    topics1, topics_perc1 = get_dominant_topic(lda, corpus_)
    topics2, topics_perc2 = get_secondary_topic(lda, corpus_)
    topics3, topics_perc3 = get_terciary_topic(lda, corpus_)

    return (
        lda,
        topics1, topics_perc1,
        topics2, topics_perc2,
        topics3, topics_perc3
    )


def genSankey(df, cat_cols=[], value_cols='', title=''):
    """Sets up a dictionary to generate a Sankey plot using Plotly."""
    # maximum of 6 value cols -> 6 colors
    colorPalette = [
        '#306998',
        '#306998',
        '#306998',
    ]
    labelList = []
    colorNumList = []
    for catCol in cat_cols:
        labelListTemp = list(set(df[catCol].values))
        colorNumList.append(len(labelListTemp))
        labelList = labelList + labelListTemp

    # remove duplicates from labelList
    labelList = list(dict.fromkeys(labelList))

    # define colors based on number of levels
    colorList = []
    for idx, colorNum in enumerate(colorNumList):
        colorList = colorList + [colorPalette[idx]]*colorNum

    # transform df into a source-target pair
    for i in range(len(cat_cols)-1):
        if i == 0:
            sourceTargetDf = df[[cat_cols[i], cat_cols[i+1], value_cols]]
            sourceTargetDf.columns = ['source', 'target', 'count']
        else:
            tempDf = df[[cat_cols[i], cat_cols[i+1], value_cols]]
            tempDf.columns = ['source', 'target', 'count']
            sourceTargetDf = pd.concat([sourceTargetDf, tempDf])
        sourceTargetDf = sourceTargetDf\
            .groupby(['source', 'target'])\
            .agg({'count': 'sum'}).reset_index()

    # add index for source-target pair
    sourceTargetDf['sourceID'] = sourceTargetDf['source'].apply(
        lambda x: labelList.index(x)
    )
    sourceTargetDf['targetID'] = sourceTargetDf['target'].apply(
        lambda x: labelList.index(x)
    )

    # creating the sankey diagram
    data = dict(
        type='sankey',
        node=dict(
          pad=15,
          thickness=20,
          line=dict(
            color="black",
            width=0.5
          ),
          label=labelList,
          color=colorList
        ),
        link=dict(
          source=sourceTargetDf['sourceID'],
          target=sourceTargetDf['targetID'],
          value=sourceTargetDf['count']
        )
      )

    layout = dict(
        title=title,
        font=dict(
          size=10
        )
    )

    fig = dict(data=[data], layout=layout)
    return fig


def set_node_community(G, communities):
    """Add community to node attributes"""
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1


def set_edge_community(G):
    """Find internal edges and add their community to their attributes"""
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0


def get_color(i, r_off=1, g_off=1, b_off=1):
    """Assign a color to a vertex."""
    n = 16
    low, high = 0.1, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)


def undirected_network_analysis(df, source_col, target_col, weights=None):
    """
    Analyses an undirected network graph.
    Parameter weights can be a list of strings.
    Exports node and edge data to produce visualizations via Gephy.
    """
    df = df[
        (df['source'] != 'data augmentation')
        &
        (df['target'] != 'data augmentation')
    ]

    G = nx.from_pandas_edgelist(df, source_col, target_col, weights)

    # Extract network data
    eigenvector_centrality = nx.eigenvector_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    betweeness_centrality = nx.betweenness_centrality(G)
    clustering_coef = nx.clustering(G)
    pagerank = nx.pagerank(G, alpha=0.85)

    # Make nodes dataframe
    df_nodes = pd.DataFrame(dict(
        eigenvector=eigenvector_centrality,
        closeness=closeness_centrality,
        betweeness=betweeness_centrality,
        clustering=clustering_coef,
        pagerank=pagerank
    ))
    df_nodes.index.name = 'Keywords'

    # Community detection
    communities = nxcom.greedy_modularity_communities(G, weight='Avg Cites')
    set_node_community(G, communities)
    set_edge_community(G)

    df_nodes['community'] = [G.nodes[node]['community'] for node in G.nodes]

    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]

    pos = nx.spring_layout(G)

    nx.draw_networkx(
        G,
        pos=pos,
        node_color=node_color,
    )
    plt.show()

    return G


if __name__ == '__main__':

    # Set up variables and basic configs
    data_path, results_path, analysis_path = generate_paths(__file__)
    results = pickle.load(open(join(results_path, 'results.pkl'), 'rb'))
    df = results['original_data']

    # Set up template formatting for plots
    load_plt_sns_configs(font_size=12)

    # Journal and conference analysis
    top_journals, top_conferences = main_venues(
        results['journal_analysis_data'],
        top_n=10
    )

    top_journals.to_csv(join(analysis_path, 'top_journals.csv'), sep=';')
    top_conferences.to_csv(join(analysis_path, 'top_conferences.csv'), sep=';')

    # Papers with most citations
    top_papers = df.sort_values('Cited by', ascending=False)[[
        'Authors', 'Title', 'Year', 'Cited by'
    ]].head(10)
    top_papers['Cited by'] = top_papers['Cited by'].astype(int)
    top_papers.to_csv(
        join(analysis_path, 'top_papers.csv'),
        index=False,
        sep=';'
    )

    # Number of publications per year
    per_year_publications(results['journal_analysis_data'])

    # LDA
    (
        lda,
        topics1, topics_perc1,
        topics2, topics_perc2,
        topics3, topics_perc3
    ) = lda_analysis(
        results['preprocessed_docs_lda'],
        num_topics=LDA_TOPICS,
        line_plot=False  # Perform the elbow method, computationally expensive
    )

    topic_keywords = {
        i+1: ', '.join(
            dict(lda.show_topic(i)).keys()
        )
        for i in range(LDA_TOPICS)
    }

    df['topic1'] = topics1
    df['topic_perc1'] = topics_perc1
    df['topic2'] = topics2
    df['topic_perc2'] = topics_perc2
    df['topic3'] = topics3
    df['topic_perc3'] = topics_perc3

    df_topics = df.groupby('topic2').apply(
        lambda dat: dat.sort_values('Cited by', ascending=False).head(1)
    ).reset_index(drop=True).set_index('topic2')[['Title']]

    df_topics['Papers'] = df.groupby('topic2').size()
    df_topics = df_topics.join(pd.Series(topic_keywords, name='Words'))
    df_topics.Words = df_topics.Words.apply(lambda x: x.replace('_','\_'))
    df_topics.index.rename('Topic', inplace=True)
    df_topics.rename(columns={'Title': 'Representative Paper'}, inplace=True)
    df_topics.to_csv(join(analysis_path, 'topic_analysis.csv'), sep=';')

    # Topic analysis
    df_sankey = df\
        .groupby(['topic1', 'topic2', 'topic3'])\
        .size().to_frame('count').reset_index()

    df_sankey.topic1 = df_sankey.topic1.astype(str) + 'a'
    df_sankey.topic2 = df_sankey.topic2.astype(str) + 'b'
    df_sankey.topic3 = df_sankey.topic3.astype(str) + 'c'

    fig = go.Figure(
        genSankey(
            df_sankey,
            cat_cols=['topic1', 'topic2', 'topic3'],
            value_cols='count'
        )
    )
    fig.write_image(
        join(analysis_path, "lda_topics_sankey.pdf")
    )

    # U-map analysis
    df['umap_x'], df['umap_y'] = results['document_umap_projections'].T

    sns.scatterplot(
        x=df.umap_x.values,
        y=df.umap_y.values,
        hue=df.topic2.values.astype(str),
        size=4
    )
    plt.legend(bbox_to_anchor = (1.0, 0.75))
    plt.savefig(
        join(analysis_path, 'umap_lda_topics.pdf'),
        bbox_inches='tight'
    )
    plt.close()

    # Per Year topic frequency
    df_topics_year = (
        df.groupby(['Year', 'topic2']).size() / df.groupby(['Year']).size()
    ).to_frame('perc').reset_index().rename(columns={'topic2': 'Topic'})
    df_topics_year.pivot('Year', 'Topic', 'perc').plot.bar(stacked=True)
    plt.legend(bbox_to_anchor = (1.0, 0.75))
    plt.savefig(join(analysis_path, 'topics_per_year.pdf'), bbox_inches='tight')
    plt.close()

    # Network analysis
    G = undirected_network_analysis(
        results['network_data'],
        'source',
        'target',
        ['Nbr of documents', 'Avg cites', 'weight']
    )
    nx.write_gexf(G, join(analysis_path, "keyword_graph.gexf"))
