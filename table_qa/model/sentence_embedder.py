from typing import List
from sentence_transformers import SentenceTransformer

def get_sentence_embedder(model_name: str = "deepset/all-mpnet-base-v2-table"):
    retriever = SentenceTransformer(model_name)
    return retriever

if __name__ == '__main__':
    from table_qa.data.dataset_load import generate_fake_data, table_preprocesser
    df = generate_fake_data(200)
    preprocessed_table = table_preprocesser([df])

    retriever = get_sentence_embedder()
