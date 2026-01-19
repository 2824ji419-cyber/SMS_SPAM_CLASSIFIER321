import nltk

resources = [('punkt', 'tokenizers/punkt'), ('punkt_tab', 'tokenizers/punkt_tab'), ('stopwords', 'corpora/stopwords')]

print("Verifying NLTK resources...")
for name, path in resources:
    try:
        nltk.data.find(path)
        print(f"[OK] {name} found.")
    except LookupError:
        print(f"[MISSING] {name} not found. Attempting download...")
        try:
            nltk.download(name)
            print(f"[OK] {name} downloaded.")
        except Exception as e:
            print(f"[ERROR] Failed to download {name}: {e}")

print("Verification complete.")
