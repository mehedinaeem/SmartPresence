import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.recognition.embedding_extractor import extract_embeddings

if __name__ == "__main__": print(extract_embeddings())
