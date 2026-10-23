# %% [markdown]
# Source notebook: Lesson_2.ipynb
# Cell order and explanatory text are preserved; saved outputs are omitted.
# See SOURCE_FILES.md at the repository root for runtime requirements.

# %% [markdown]
# # L2: Build Customized RAG

# %% [markdown]
# <p style="background-color:#fff6e4; padding:15px; border-width:3px; border-color:#f5ecda; border-style:solid; border-radius:6px"> ⏳ <b>Note <code>(Kernel Starting)</code>:</b> This notebook takes about 30 seconds to be ready to use. You may start and watch the video while you wait.</p>

# %%
# Notebook cell 3
import os
import warnings
from helper import load_env

warnings.filterwarnings('ignore')
load_env()

# %%
# Notebook cell 4
from haystack import Pipeline
from haystack.utils.auth import Secret
from haystack.components.builders import PromptBuilder
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators import OpenAIGenerator
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder, CohereTextEmbedder

# %% [markdown]
# <p style="background-color:#fff6ff; padding:15px; border-width:3px; border-color:#efe6ef; border-style:solid; border-radius:6px"> 💻 &nbsp; <b>Access <code>requirements.txt</code> and <code>helper.py</code> files:</b> 1) click on the <em>"File"</em> option on the top menu of the notebook and then 2) click on <em>"Open"</em>. For more help, please see the <em>"Appendix - Tips and Help"</em> Lesson.</p>

# %% [markdown]
# - Fetch Contents from URLs with [`LinkContentFetcher`](https://docs.haystack.deepset.ai/docs/linkcontentfetcher?utm_campaign=developer-relations&utm_source=dlai)
# - Convert them to Documents with [`HTMLToDocument`](https://docs.haystack.deepset.ai/docs/htmltodocument?utm_campaign=developer-relations&utm_source=dlai)
# - Create embeddings for them with [`CohereDocumentEmbedder`](https://docs.haystack.deepset.ai/docs/coheredocumentembedder?utm_campaign=developer-relations&utm_source=dlai)
# - Write them to an [`InMemoryDocumentStore`](https://docs.haystack.deepset.ai/docs/inmemorydocumentstore?utm_campaign=developer-relations&utm_source=dlai)
# 
# > ℹ️ Model providers may have outages. If you encounter issues creating embeddings or generating responses, feel free to consider any of the other [Embedder](https://docs.haystack.deepset.ai/docs/embedders?utm_campaign=developer-relations&utm_source=dlai) or [Generator](https://docs.haystack.deepset.ai/docs/generators?utm_campaign=developer-relations&utm_source=dlai) options. For this lesson, we recomment Cohere embedders, or small [Sentence Transformers](https://docs.haystack.deepset.ai/docs/sentencetransformersdocumentembedder?utm_campaign=developer-relations&utm_source=dlai) embedders.

# %% [markdown]
# ## Indexing Documents

# %%
# Notebook cell 8
document_store = InMemoryDocumentStore()

fetcher = LinkContentFetcher()
converter = HTMLToDocument()
embedder = CohereDocumentEmbedder(model="embed-english-v3.0", api_base_url=os.getenv("CO_API_URL"))
writer = DocumentWriter(document_store=document_store)

indexing = Pipeline()
indexing.add_component("fetcher", fetcher)
indexing.add_component("converter", converter)
indexing.add_component("embedder", embedder)
indexing.add_component("writer", writer)

indexing.connect("fetcher.streams", "converter.sources")
indexing.connect("converter", "embedder")
indexing.connect("embedder", "writer")

# %%
# Notebook cell 9
indexing.show()

# %%
# Notebook cell 10
# # This removes all documents from the store
# document_store.delete_documents()

# %%
# Notebook cell 11
indexing.run(
    {
        "fetcher": {
            "urls": [
                "https://haystack.deepset.ai/integrations/cohere",
                "https://haystack.deepset.ai/integrations/anthropic",
                "https://haystack.deepset.ai/integrations/jina",
                "https://haystack.deepset.ai/integrations/nvidia",
                "https://haystack.deepset.ai/integrations/azure",
                "https://haystack.deepset.ai/integrations/chroma-documentstore",
                "https://haystack.deepset.ai/integrations/deepeval",
                "https://haystack.deepset.ai/integrations/elevenlabs",
                "https://haystack.deepset.ai/integrations/elasticsearch-document-store",
                "https://haystack.deepset.ai/integrations/google-ai",
                "https://haystack.deepset.ai/integrations/github",
                "https://haystack.deepset.ai/integrations/google-genai",
                "https://haystack.deepset.ai/integrations/google-vertex-ai",
                "https://haystack.deepset.ai/integrations/huggingface",
                "https://haystack.deepset.ai/integrations/llama_cpp",
                "https://haystack.deepset.ai/integrations/milvus-document-store",
                "https://haystack.deepset.ai/integrations/mistral",
                "https://haystack.deepset.ai/integrations/neo4j-document-store",
                "https://haystack.deepset.ai/integrations/ollama",
                "https://haystack.deepset.ai/integrations/pinecone-document-store",
                "https://haystack.deepset.ai/integrations/qdrant-document-store",
                "https://haystack.deepset.ai/integrations/snowflake",
                "https://haystack.deepset.ai/integrations/weaviate-document-store",
                "https://haystack.deepset.ai/integrations/weights-and-bias-tracer"
            ]
        }
    }
)

# %%
# Notebook cell 12
document_store.filter_documents()[0]

# %% [markdown]
# ## Retrieval Augmented Generation
# ### 1. Decide on the Prompt
# Augment the prompt with the contents of these documents using the [`PromptBuilder`](https://docs.haystack.deepset.ai/docs/promptbuilder?utm_campaign=developer-relations&utm_source=dlai). This component uses Jinja templating [[+]](https://jinja.palletsprojects.com/en/3.1.x/)

# %%
# Notebook cell 14
prompt = """
Answer the question based on the provided context.
Context:
{% for doc in documents %}
   {{ doc.content }} 
{% endfor %}
Question: {{ query }}
"""

# %% [markdown]
# ### 2. Build the Pipeline

# %%
# Notebook cell 16
query_embedder = CohereTextEmbedder(model="embed-english-v3.0", api_base_url=os.getenv("CO_API_URL"))
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt)
generator = OpenAIGenerator()

rag = Pipeline()
rag.add_component("query_embedder", query_embedder)
rag.add_component("retriever", retriever)
rag.add_component("prompt", prompt_builder)
rag.add_component("generator", generator)

rag.connect("query_embedder.embedding", "retriever.query_embedding")
rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt", "generator")

# %% [markdown]
# > Note: It is possible to use a different model for the generator. For example, if you'd like to use Llama-3, update the code above to:
# 
# ```
# generator = OpenAIGenerator(api_key=Secret.from_env_var("TOGETHER_AI_API"),
#                             model="meta-llama/Llama-3-70b-chat-hf",
#                             api_base_url="https://api.together.xyz/v1")
# ```

# %%
# Notebook cell 18
rag.show()

# %%
# Notebook cell 19
question = "How can I use Cohere with Haystack?"

result = rag.run(
    {
        "query_embedder": {"text": question},
        "retriever": {"top_k": 1},
        "prompt": {"query": question},
    }
)

print(result["generator"]["replies"][0])

# %% [markdown]
# ### 3. Customize The Behaviour

# %%
# Notebook cell 21
prompt = """
You will be provided some context, followed by the URL that this context comes from.
Answer the question based on the context, and reference the URL from which your answer is generated.
Your answer should be in {{ language }}.
Context:
{% for doc in documents %}
   {{ doc.content }} 
   URL: {{ doc.meta['url']}}
{% endfor %}
Question: {{ query }}
Answer:
"""

# %%
# Notebook cell 22
query_embedder = CohereTextEmbedder(model="embed-english-v3.0", api_base_url=os.getenv("CO_API_URL"))
retriever = InMemoryEmbeddingRetriever(document_store=document_store)
prompt_builder = PromptBuilder(template=prompt)
generator = OpenAIGenerator(model="gpt-3.5-turbo")

rag = Pipeline()
rag.add_component("query_embedder", query_embedder)
rag.add_component("retriever", retriever)
rag.add_component("prompt", prompt_builder)
rag.add_component("generator", generator)

rag.connect("query_embedder.embedding", "retriever.query_embedding")
rag.connect("retriever.documents", "prompt.documents")
rag.connect("prompt", "generator")

# %%
# Notebook cell 23
question = "How can I use Cohere with Haystack?"

result = rag.run(
    {
        "query_embedder": {"text": question},
        "retriever": {"top_k": 1},
        "prompt": {"query": question, "language": "French"},
    }
)

print(result["generator"]["replies"][0])

# %%
# Notebook cell 24


# %%
# Notebook cell 25


# %%
# Notebook cell 26


# %%
# Notebook cell 27


# %%
# Notebook cell 28


# %%
# Notebook cell 29

