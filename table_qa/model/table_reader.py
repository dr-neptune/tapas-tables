from transformers import pipeline, TapasTokenizer, TapasForQuestionAnswering

def load_tapas(model_name: str = "google/tapas-base-finetuned-wtq"):
    tokenizer = TapasTokenizer.from_pretrained(model_name)
    model = TapasForQuestionAnswering.from_pretrained(model_name, local_files_only=False)
    pipe = pipeline("table-question-answering",  model=model, tokenizer=tokenizer)
    return pipe
