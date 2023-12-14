# Documentation

## Index

- [Overview](#overview)
  - [Notes](#notes)
- [Modules](#modules)
  - [Models](models.md)
  - [Embeddings](embeddings.md)
  - [Memory](memory.md)
  - [Vector Stores](vectorStores.md)
  - [Cache](cache.md)
  - [Chatbots](chatbots/README.md)
    - [Chatbot](chatbots/chatbot.md)
    - [Document Chatbot](chatbots/documentChatbot.md)
  - [Exceptions](exceptions.md)
  - [Schemas](schemas/README.md)
    - [Message](schemas/message.md)
    - [Usage](schemas/usage.md)
    - [Response](schemas/response.md)
    - [Filter](schemas/filter.md)
    - [OpenAI Chat Choice](schemas/openAIChatChoice.md)
    - [OpenAI Chat Response](schemas/openAIChatResponse.md)
    - [Vector](schemas/vector.md)

## Overview

This package contains a set of tools that allow you to quickly develop applications that use LLMs (Large Language Models), particularly those related to chatbots.

It has an assortment of classes that will help you implement functionality such as interaction with LLMs; response caching for improved performance and saving tokens; interfaces with vector databases; dynamic conversation history management and more.

### Notes:

- This package was built using **Python** version `3.10.11`. We strongly recommend using the same version within your virtual environment.
- The current version of the package is still under development (it’s in its Alpha stage), so a few features may seem somewhat limited. However, inheriting from the base classes and expanding on the current functionality is highly encouraged.

**We highly recommend to start with the [Chatbots](./chatbots/README.md) section, as it is the heart of the package.**

## Modules

- [Models](models.md)
- [Embeddings](embeddings.md)
- [Memory](memory.md)
- [Vector Stores](vectorStores.md)
- [Cache](cache.md)
- [Chatbots](chatbots/README.md)
  - [Chatbot](chatbots/chatbot.md)
  - [Document Chatbot](chatbots/documentChatbot.md)
- [Exceptions](exceptions.md)
- [Schemas](schemas/README.md)
  - [Message](schemas/message.md)
  - [Usage](schemas/usage.md)
  - [Response](schemas/response.md)
  - [Filter](schemas/filter.md)
  - [OpenAI Chat Choice](schemas/openAIChatChoice.md)
  - [OpenAI Chat Response](schemas/openAIChatResponse.md)
  - [Vector](schemas/vector.md)
