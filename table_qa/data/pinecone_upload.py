import pinecone
import os
from typing import List, Dict, Any
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from table_qa.model.sentence_embedder import SentenceTransformer
load_dotenv()


def connect_to_pinecone() -> None:
    pinecone.init(api_key=os.environ.get("POETRY_PINECONE_API_KEY"),
                  environment=os.environ.get("POETRY_PINECONE_ENVIRONMENT"))
    print('Connected to Pinecone')


def create_index(index_name: str, dims: int, metric: str = "cosine") -> pinecone.Index:
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dims, metric=metric)

    index = pinecone.Index(index_name)
    return index


def load_tables_into_pinecone(pinecone_index: pinecone.Index, processed_tables: List[str], batch_size: int = 64) -> None:
    for i in tqdm(range(0, len(processed_tables), batch_size)):
        # find end of batch
        i_end = min(i+batch_size, len(processed_tables))
        # extract batch
        batch = processed_tables[i:i_end]
        # generate embeddings for batch
        emb = retriever.encode(batch).tolist()
        # create unique IDs ranging from zero to the total number of tables in the dataset
        ids = [f"{idx}" for idx in range(i, i_end)]
        # add all to upsert list
        to_upsert = list(zip(ids, emb))
        # upsert/insert these records to pinecone
        _ = pinecone_index.upsert(vectors=to_upsert)

    # check that we have all vectors in index
    print(pinecone_index.describe_index_stats())


def query_pinecone(query: str,
                   pinecone_index: pinecone.Index,
                   retriever: SentenceTransformer) -> Dict[str, Any]:
    xq = retriever.encode([query]).tolist()
    result = pinecone_index.query(xq, top_k=2)
    return result

result_table = query_pinecone("Which company has the highest total value?", pinecone_index, retriever)

def get_answer_from_table(table, query):
    answers = load_tapas()(table, query)
    return answers

get_answer_from_table(tables[0], "Which company has the highest total value?")


# try again!




if __name__ == "__main__":
    connect_to_pinecone()

    # get embedding dimension
    from table_qa.model.sentence_embedder import get_sentence_embedder
    retriever = get_sentence_embedder()
    dims = retriever.get_sentence_embedding_dimension()

    # create index
    pinecone_index = create_index("table-qa", dims, "cosine")

    # load tables into pinecone
    from table_qa.data.dataset_load import generate_fake_data, table_preprocesser
    tables = [generate_fake_data(1000)]

    processed_tables = table_preprocesser(tables)

    load_tables_into_pinecone(pinecone_index, processed_tables)

    query = "Which company has the highest total value?"
    query = "Which industry has the highest total_value?"

    xq = retriever.encode([query]).tolist()
    result = pinecone_index.query(xq, top_k=2)

    tbl_id = int(result['matches'][0]['id'])
    tables[tbl_id]

    from table_qa.model.table_reader import load_tapas
    tapas = load_tapas()

    results = tapas(table=tables[tbl_id], query=query)

    results = tapas(table=[df], query=query)

    results = tapas({'table': df}, query=query)

    print(pd.DataFrame({'table': df}['table']))

    pd.DataFrame(str(df.to_dict()))

    str(df.to_dict())



    pd.DataFrame(df.to_string())

    from io import StringIO

    pd.DataFrame(pd.read_csv(StringIO(tables[tbl_id]), sep=','))

    # try again
    from gapminder import gapminder

    tables = [gapminder]

    tables = dfs

    retriever = get_sentence_embedder()
    dims = retriever.get_sentence_embedding_dimension()

    processed_tables = table_preprocesser(tables)

    pinecone_index = create_index("gapminder", dims, "cosine")

    load_tables_into_pinecone(pinecone_index, processed_tables)

    query = "Which country has the highest life expectancy?"
    query = "Which country has the highest gdpPercap?"

    xq = retriever.encode([query]).tolist()

    result = pinecone_index.query(xq, top_k=2)

    tbl_id = int(result['matches'][0]['id'])
    tbl_id = 16

    tapas = load_tapas()

    print(gapminder.loc[gapminder['gdpPercap'].idxmax()])

    results = tapas(table=tables[tbl_id].reset_index().astype(str), query=query)

    results = tapas(table=tables[tbl_id].astype(str)[:20], query=query)

    print(results)

    print(tables[tbl_id].iloc[19,])

    tables[tbl_id][:20]

    query = "What year has the highest life expectancy?"
    results = tapas(table=tables[tbl_id].astype(str)[:20], query=query)

    query = "What is the average life expectancy for Albania?"
    results = tapas(table=tables[tbl_id].astype(str)[:200], query=query)

    # heuristic
    # chunk into 80 rows a piece
    # then search across the chunks
    import pandas as pd

    # Create a sample DataFrame with 1000 rows
    df = gapminder

    # Break the DataFrame into a list of DataFrames with 80 rows each
    chunk_size = 80
    dfs = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Print the number of DataFrames and the shape of each DataFrame
    print(f"Number of DataFrames: {len(dfs)}")
    for i, chunk in enumerate(dfs, 1):
        print(f"DataFrame {i}: {chunk.shape}")
