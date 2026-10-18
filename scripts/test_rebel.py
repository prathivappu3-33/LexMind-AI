from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "Babelscape/rebel-large"

print("Loading REBEL...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

print("Model Loaded!\n")


texts = [
    "Docker Hub supports multi-factor authentication.",
    "Git uses SHA-1 hashing.",
    "Kubernetes uses etcd as its distributed key-value store.",
    "ROS2 uses DDS for communication.",
    "Amazon Web Services provides EC2 and S3."
]


for text in texts:

    print("=" * 80)
    print("INPUT")
    print(text)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    generated_tokens = model.generate(
        **inputs,
        max_length=256,
    )

    decoded = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=False,
    )

    print("\nRAW OUTPUT")
    print(decoded[0])