# Chatbots

This module is the heart of the package. It contains the different chatbot classes that can be used to interact with the models. Chatbots are capable of holding conversations, storing and retrieving from cache and even chatting with documents.

### Index

- [**Chatbot**](chatbot.md). The Chatbot class is the main class of the library. It is used to initialize a chatbot instance, which can then be used to chat with an LLM.
- [**Document Chatbot**](documentChatbot.md). A chatbot that uses a knowledge base to answer questions. The knowledge base is a vector store that contains the documents. The embeddings model is used to embed the documents and the prompt. The model is used to generate the response.
