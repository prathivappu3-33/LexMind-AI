from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Babelscape/rebel-large"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model loaded successfully!")