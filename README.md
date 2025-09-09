# Building AI Applications With Haystack

This repository contains the **notebooks and hands-on materials** for the course **"Building AI Applications With Haystack"**, provided by **[DeepLearning.AI](https://www.deeplearning.ai/)** in collaboration with **[Haystack](https://haystack.deepset.ai/)**.

The course teaches you how to use **Haystack**, an open-source framework, to build powerful **retrieval-augmented generation (RAG) pipelines**, intelligent agents, and custom AI applications.

---

## 📚 Course Topics

### 1️⃣ Haystack Building Blocks

In this lesson, you’ll explore the **fundamental building blocks** of Haystack:

* **Components** – the modular units of computation (retrievers, readers, generators, etc.).
* **Running Components Individually** – test each piece before combining.
* **Combining Components** – link them together to form data flows.
* **Pipelines** – construct workflows for different AI tasks.
* **Document Search Pipeline** – build your first simple search system.

This provides a solid foundation for understanding how **everything in Haystack fits together**.

---

### 2️⃣ Build Customized RAG

Here, you’ll go beyond the basics and **create your own Retrieval-Augmented Generation (RAG) pipeline**:

* Combine retrievers, readers, and LLMs into a **RAG workflow**.
* Customize the **behavior of RAG** (e.g., change how many documents to retrieve, tweak prompts).
* Understand how to **adapt RAG to different tasks**, such as answering factual questions or summarizing documents.

By the end, you’ll be comfortable designing **tailored RAG pipelines** for your own applications.

---

### 3️⃣ Custom Components – News Summarizer

This lab shows you how to **extend Haystack** by writing custom components:

* Build your **own custom component** in Python.
* Create a **News Summarizer component** that condenses long news articles into short, useful summaries.
* Integrate your custom component into a pipeline, mixing it with existing Haystack modules.

You’ll learn how to **go beyond out-of-the-box features** and craft components for **specialized use cases**.

---

### 4️⃣ Fallback with Branching Pipeline

Sometimes AI systems **fail or produce empty results**.
In this lesson, you’ll learn to handle that gracefully:

* Use **branching pipelines** to add decision points.
* Implement **fallback logic** (e.g., if a retriever fails, fall back to a different one).
* Improve system reliability by **ensuring robust outputs even in edge cases**.

This prepares you for **real-world scenarios** where robustness is essential.

---

### 5️⃣ Self-Reflecting Agents with Loops

Agents can **think about their own outputs and refine them**.
In this lesson, you’ll build **self-reflecting agents**:

* Implement **feedback loops** where the agent checks its own answers.
* Learn how looping helps **reduce hallucinations** and **improve accuracy**.
* Explore **advanced agent workflows** for complex tasks.

This introduces a **more autonomous and intelligent way of building AI agents**.

---

### 6️⃣ Chat Agent with Function Calling

Finally, you’ll bring everything together by building an **interactive AI agent**:

* Create a **chat agent** that can have back-and-forth conversations.
* Use **function calling** to let the agent **trigger tools** (like search, summarization, or APIs).
* Build an AI that doesn’t just answer questions, but can **act dynamically**.

This is the **capstone skill** for building **real AI assistants** with Haystack.

---

## 🙌 Acknowledgements

This course and its content are developed by **DeepLearning.AI** in collaboration with **Haystack (deepset)**.
All credits for teaching and materials go to them - this repo simply organizes the **notebooks for easy access and learning**.

---

✨ *With these lessons, you’ll gain the skills to build robust, customized, and intelligent AI applications using Haystack.*
