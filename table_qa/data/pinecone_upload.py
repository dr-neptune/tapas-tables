import pinecone
import os
from typing import List
from tqdm import tqdm

from dotenv import load_dotenv
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
    df = generate_fake_data(1000)

    tables = table_preprocesser(df)

    load_tables_into_pinecone(pinecone_index, tables)

    query = "Which company has the highest total value?"
    query = "Which industry has the highest total_value?"

    xq = retriever.encode([query]).tolist()
    result = pinecone_index.query(xq, top_k=2)

    tbl_id = int(result['matches'][0]['id'])
    tables[tbl_id]

    from table_qa.model.table_reader import load_tapas
    tapas = load_tapas()

    results = tapas(table={'table': [tables[tbl_id]]}, query=query)

    StringIO
