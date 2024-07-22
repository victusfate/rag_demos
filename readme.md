# RAG System Demo: Interactive Text Analysis with AI

## Overview

This project demonstrates a Retrieval-Augmented Generation (RAG) system that combines web scraping, text embedding, and AI-powered question answering. It's designed to fetch text from a web source, process it, and answer questions based on the content using advanced natural language processing techniques.

## Features

- Web scraping to fetch text from specified URLs
- Text processing and chunking for efficient analysis
- Vector store implementation for semantic search
- Integration with the Microsoft Phi-2 language model for question answering
- Interactive query interface

## Requirements

- Python 3.7+
- Google Colab environment (for easy setup and GPU access)

## Installation

This project is designed to run in Google Colab. To use it:

1. Open the notebook in Google Colab
2. Run the first cell to install required packages:
   ```
   !pip install requests beautifulsoup4 torch transformers sentence-transformers numpy scikit-learn
   ```
3. Run subsequent cells to set up the environment and initialize the models

A command line version has also been added for local cli usage.

## Usage

After setting up the environment and running the initialization cells:

1. The system will automatically fetch and process text from a specified URL (default: Project Gutenberg's "The Ten Commandments")
2. Use the `interactive_query()` function to ask questions about the text

Example query and response:

```python
query = "List the ten commandments"
relevant_context = vector_store.search(query)
response = generate_response(query, relevant_context)
print(f"Query: {query}")
print(f"Response: {response}")
```

Output:
```
Query: List the ten commandments
Response: 1. Thou shalt not kill. 2. Thou shalt not commit adultery. 3. Thou shalt not steal. 4. Thou shalt not bear false witness against thy neighbour. 5. Thou shalt not covet thy neighbour's house, thou shalt not covet thy neighbour's wife, nor thy neighbour's manservant, nor thy neighbour's ox, nor thy neighbour's ass, nor thy neighbour's sheep, nor thy neighbour's oxen, nor thy neighbour's donkey, nor thy neighbour's mule, nor thy neighbour's ass, nor thy neighbour's oxen, nor thy neighbour's donkey, nor thy neighbour's mule. 6. Thou shalt not covet thy neighbour's wife, nor thy neighbour's manservant, nor thy neighbour's ox, nor thy neighbour's ass, nor thy neighbour's sheep, nor thy neighbour's oxen, nor thy neighbour's donkey, nor thy neighbour's mule, nor thy neighbour's ass, nor thy neighbour's oxen, nor thy neighbour's donkey, nor thy neighbour's mule. 7. Thou shalt not covet thy neighbour's house, thou shalt not covet thy neighbour's wife, nor thy neighbour's manservant, nor thy neighbour's ox, nor thy
```

Note: The response is truncated due to token limitations in the model output.

## Customization

- To analyze different texts, change the `url` variable to point to a different web source
- Adjust the `chunk_size` in the `SimpleVectorStore` class to optimize for different text lengths
- Modify the `generate_response` function to alter the system prompt or response generation parameters

## Limitations

- The system's knowledge is limited to the text it has processed from the given URL
- Response quality may vary depending on the complexity of the query and the relevance of the retrieved context
- The model may occasionally produce incomplete or inaccurate responses, especially for complex queries


## Contributing

Contributions to improve the system are welcome. Please feel free to fork the repository, make changes, and/or submit pull requests.

## License

This project is open-source and available under the MIT License.