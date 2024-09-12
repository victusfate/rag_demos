from haystack import Pipeline
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack import Document

docstore = InMemoryDocumentStore()
docstore.write_documents([
    Document(content="My name is Jean and I live in Paris."), 
    Document(content="My name is Mark and I live in Berlin."), 
    Document(content="My name is Giorgio and I live in Rome.")
])

generator = HuggingFaceLocalGenerator(model="google/flan-t5-large",
                                      task="text2text-generation",
                                      generation_kwargs={
                                        "max_new_tokens": 100,
                                        "temperature": 0.9,
                                        })

# query = "Who lives in Paris?"
query = "Who lives in Paris?"

template = """
Given the following information, answer the question.

Context: 
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{ query }}?
"""
pipe = Pipeline()

pipe.add_component("retriever", InMemoryBM25Retriever(document_store=docstore))
pipe.add_component("prompt_builder", PromptBuilder(template=template))
pipe.add_component("llm", generator)
pipe.connect("retriever", "prompt_builder.documents")
pipe.connect("prompt_builder", "llm")

res=pipe.run({
    "prompt_builder": {
        "query": query
    },
    "retriever": {
        "query": query
    }
})

print(res)