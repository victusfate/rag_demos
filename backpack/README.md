# Backpack

Backpack is a RAG (Retrieval-Augmented Generation) application for local document processing and querying. This application allows users to process text documents from a directory, store their content in a SQLite database, create embeddings using a sentence transformer model, and perform similarity searches to answer queries based on the document content.

## Features

- Process text documents from a specified directory
- Store document content in a SQLite database
- Generate embeddings for document content using the `sentence-transformers` library
- Perform similarity searches using `faiss`
- Answer queries based on the most relevant documents using a pre-trained language model

## Installation

### Prerequisites

- Python 3.11 or higher
- Poetry (https://python-poetry.org/)

### Steps

1. **Clone the repository:**

   ```sh
   git clone https://github.com/victusfate/rag_demos.git
   cd backpack
