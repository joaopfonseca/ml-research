"""
Data preprocessing
"""
from os.path import join
import pandas as pd
from sqlite3 import connect
from research.utils import (
    generate_paths,
    load_datasets
)

DATA_PATH = generate_paths(__file__)[0]

if __name__ == '__main__':

    # Read and concatenate data
    data = load_datasets(
        DATA_PATH,
        suffix='.csv',
        target_exists=False,
        error_bad_lines=False
    )
    df = pd.concat([dat for _, dat in data]).dropna(subset=['DOI', 'Cited by'])
    df_journals = pd.concat([dat for _, dat in data]).dropna(subset=['DOI'])

    # Select relevant columns
    columns = [
        'Authors',
        'Author(s) ID',
        'Title',
        'Year',
        'Cited by',  # Number of citations
        'DOI',
        'Affiliations',
        'Abstract',
        'Author Keywords',
        'Source title',  # Journal/Conference where the paper was published
        'Conference name',
        'Publisher'
    ]

    df = df[columns]
    df_journals = df_journals[columns]

    df_journals['is_conference'] = ~df_journals['Conference name'].isna()

    # Store data into a SQLite database
    with connect(join(DATA_PATH, 'literature_preprocessed.db')) as connection:
        df.to_sql(
            'slr',
            connection,
            index=False,
            if_exists='replace'
        )
        df_journals.to_sql(
            'journals',
            connection,
            index=False,
            if_exists='replace'
        )
