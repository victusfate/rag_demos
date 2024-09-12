import os
import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import sqlite3
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
from tqdm import tqdm

class RAGApp:
    def __init__(self, master):
        self.master = master
        master.title("RAG Application")
        master.geometry("800x600")

        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)  # 384 is the dimension of all-MiniLM-L6-v2 embeddings
        
        self.conn = sqlite3.connect('documents.db')
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents
            (id INTEGER PRIMARY KEY, path TEXT, content TEXT)
        ''')
        self.conn.commit()

        # Initialize language model for query answering
        model_name = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.lm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        self.create_widgets()

    def create_widgets(self):
        self.notebook = ttk.Notebook(self.master)
        self.notebook.pack(expand=True, fill='both')

        # Data Processing Tab
        self.process_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.process_frame, text='Process Data')

        self.label = tk.Label(self.process_frame, text="Select a directory to process:")
        self.label.pack(pady=10)

        self.button = tk.Button(self.process_frame, text="Choose Directory", command=self.choose_directory)
        self.button.pack()

        self.progress = ttk.Progressbar(self.process_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(pady=10)

        self.status_label = tk.Label(self.process_frame, text="")
        self.status_label.pack()

        # Query Tab
        self.query_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.query_frame, text='Query')

        self.query_label = tk.Label(self.query_frame, text="Enter your question:")
        self.query_label.pack(pady=10)

        self.query_entry = tk.Entry(self.query_frame, width=70)
        self.query_entry.pack()

        self.query_button = tk.Button(self.query_frame, text="Ask", command=self.ask_question)
        self.query_button.pack(pady=10)

        self.result_text = scrolledtext.ScrolledText(self.query_frame, height=20, width=80)
        self.result_text.pack(pady=10)

    def choose_directory(self):
        directory = filedialog.askdirectory()
        if directory:
            self.process_directory(directory)

    def process_directory(self, directory):
        file_paths = [os.path.join(root, file) 
                      for root, _, files in os.walk(directory) 
                      for file in files if file.endswith('.txt')]
        
        total_files = len(file_paths)
        self.progress["maximum"] = total_files
        
        for i, file_path in enumerate(file_paths):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Store in SQLite
            self.cursor.execute('INSERT INTO documents (path, content) VALUES (?, ?)', (file_path, content))
            doc_id = self.cursor.lastrowid
            self.conn.commit()

            # Create and store embedding
            embedding = self.model.encode(content)
            self.index.add(np.array([embedding]))

            # Update progress
            self.progress["value"] = i + 1
            self.status_label.config(text=f"Processing: {i+1}/{total_files}")
            self.master.update_idletasks()

        self.status_label.config(text=f"Processed {total_files} files from directory: {directory}")

    def ask_question(self):
        query = self.query_entry.get()
        if query:
            self.status_label.config(text="Searching...")
            self.master.update_idletasks()

            # Perform similarity search
            query_embedding = self.model.encode(query)
            D, I = self.index.search(np.array([query_embedding]), k=2)  # Find top 2 similar documents

            # Retrieve context
            context = []
            for idx in I[0]:
                self.cursor.execute('SELECT content FROM documents WHERE id = ?', (idx + 1,))
                result = self.cursor.fetchone()
                if result:
                    context.append(result[0])

            # Generate response
            response = self.generate_response(query, context)
            
            # Display result
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, f"Query: {query}\n\nResponse: {response}")
            
            self.status_label.config(text="Search completed")

    def generate_response(self, query, context):
        system_prompt = "You are a helpful AI assistant. Provide concise and accurate answers based on the given context."
        query_wrapper_prompt = "Context: {context}\nQuestion: {query}\nAnswer:"
        
        prompt = query_wrapper_prompt.format(context=" ".join(context), query=query)
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.lm_model.device)
        
        with torch.no_grad():
            outputs = self.lm_model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=0.0
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Answer:")[-1].strip()

    def __del__(self):
        self.conn.close()

root = tk.Tk()
app = RAGApp(root)
root.mainloop()