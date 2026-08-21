DATASET_PATH = "./data/mrc-filtered.jsonl" # mrc, ARCD 
CHUNK_STRATEGY = "recursive" # Options:  "fixed", "recursive", "semantic"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
MIN_CHUNK_SIZE = 75
SIMILARITY_THRESHOLD = 0.65
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DB_PATH = "./db/MRC" # MRC, ARCD 
COLLECTION_NAME = f"mrc_collection_chunks_{CHUNK_STRATEGY}" # mrc, ARCD 
USE_NER = True
# Generation model
MODEL_NAME = "aya:8b" # ["llama3:8b","aya:8b","qwen2.5:3b","command-r7b-arabic"]
