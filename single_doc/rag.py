import sys
import subprocess
import pkg_resources

required_packages = {
    'requests': 'requests',
    'beautifulsoup4': 'bs4',
    'torch': 'torch',
    'transformers': 'transformers',
    'sentence-transformers': 'sentence_transformers',
    'numpy': 'numpy',
    'scikit-learn': 'sklearn',
    'accelerate': 'accelerate',
    'huggingface-hub': 'huggingface-hub'
}

def install_missing_packages():
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}
    print('installed packages:', installed_packages)
    missing_packages = [pkg for pkg, import_name in required_packages.items() if pkg.lower() not in installed_packages]
    print('missing packages:', missing_packages)
    if missing_packages:
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("Installation complete. Please restart the script.")
        sys.exit()

install_missing_packages()

try:
    import requests
    from bs4 import BeautifulSoup
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import accelerate
    import huggingface_hub
except ImportError as e:
    print(f"Error importing required library: {e}")
    print("Please ensure all required packages are installed correctly.")
    sys.exit(1)

def fetch_and_process_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    paragraphs = soup.find_all('p')
    text = ' '.join([p.get_text() for p in paragraphs])
    
    text = text.replace('\n', ' ').replace('\r', '')
    return text

class SimpleVectorStore:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.vectors = []
        self.texts = []

    def add_text(self, text, chunk_size=256):
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        embeddings = self.encoder.encode(chunks)
        self.vectors.extend(embeddings)
        self.texts.extend(chunks)

    def search(self, query, k=2):
        query_vector = self.encoder.encode([query])
        similarities = cosine_similarity(query_vector, self.vectors)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.texts[i] for i in top_k_indices]

def generate_response(query, context, model, tokenizer):
    system_prompt = "You are a helpful AI assistant. Provide concise and accurate answers based on the given context."
    query_wrapper_prompt = "Context: {context}\nQuestion: {query}\nAnswer:"
    
    prompt = query_wrapper_prompt.format(context=" ".join(context), query=query)
    full_prompt = f"{system_prompt}\n\n{prompt}"
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()

def interactive_query(vector_store, model, tokenizer):
    while True:
        query = input("Enter your question (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        relevant_context = vector_store.search(query)
        response = generate_response(query, relevant_context, model, tokenizer)
        print(f"\nQuery: {query}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    print("Initializing RAG system...")
    
    model_name = "microsoft/phi-2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Error loading the model: {e}")
        print("Attempting to load the model without device mapping...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16
        )
        model = model.to("cuda" if torch.cuda.is_available() else "cpu")

    vector_store = SimpleVectorStore()

    url = 'https://www.gutenberg.org/cache/epub/10/pg10.txt'
    print(f"Fetching and processing text from {url}")
    text = fetch_and_process_text(url)
    vector_store.add_text(text)

    print("System initialized. Starting interactive query session.")
    interactive_query(vector_store, model, tokenizer)

    print("Thank you for using the RAG system. Goodbye!")