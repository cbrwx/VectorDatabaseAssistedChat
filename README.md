![f323f](https://github.com/cbrwx/VectorDatabaseAssistedChat/assets/81207429/06fe406c-eaf4-4aaf-8061-5461c3639a83)
# OdinAI: A Context-Aware Chat System with Vector Database Memory

This repository contains the implementation of OdinAI, a context-aware chat system that integrates a vector database for enhancing the contextual understanding and response quality of a chat interface. OdinAI leverages the power of sentence embeddings, hierarchical clustering, and direct interaction with Large Language Models (LLMs), providing another approach to chat-based interactions.

## System Overview

### Components

- **Vector Database (`SimpleVectorDatabase`)**: Encodes chat messages into high-dimensional vectors and stores them for contextual retrieval, playing a critical role in understanding and responding to user inputs effectively.
- **Context Extraction**: Enriches messages with contextual information before encoding, utilizing external LLMs to identify relevant context, making the chat experience more relevant and personalized.
- **LLM Communication**: Direct API calls to LLMs facilitate generating responses based on the current context and the history stored in the vector database.
- **Interactive Interface**: An interactive chat environment created using IPython widgets within Jupyter Notebooks, enhancing user interaction and engagement.

### Key Features

- **Message Encoding**: Textual messages are converted into vector representations using `SentenceTransformer`. This process allows for semantic comparison and retrieval of contextually similar messages.
- **Hierarchical Clustering**: Organizes encoded messages into a hierarchical structure, enabling the efficient retrieval of similar messages and insights into the conversation's thematic organization.
- **Direct LLM Interaction**: Streamlined communication with LLMs enables the system to generate contextually aware responses, improving the chat system's effectiveness.
- **Dynamic Context Understanding**: The system continuously updates the vector database with new interactions, enhancing its ability to understand and respond to users' queries over time.

## Technical Description

### Vector Database (`SimpleVectorDatabase`)

- **Initialization (`__init__`)**: Sets up the vector database, supporting optional persistence through a file path and defining a clustering threshold. It initializes internal structures for storing vectors, messages, types, and contexts, including mechanisms for clustering and context caching.
- **Context Extraction (`extract_context`)**: Contacts an LLM to extract context from input messages, enhancing them with additional semantic information before encoding. This process involves an API call to a configured LLM endpoint.
- **Message Encoding (`encode_message`)**: Converts messages into dense vector representations, considering extracted context for comprehensive encoding. It handles chunking to accommodate model limitations.
- **Vector Addition (`add_vector`)**: Adds new vectors to the database, avoiding duplicates, and optionally extracts context if not provided. Updates clusters and persists changes if applicable.
- **Interaction Recording (`add_interaction`)**: Records query-response interactions by encoding both and adding them to the database, crucial for building historical context.
- **Clustering Threshold Adjustment (`adjust_clustering_threshold`)**: Allows adjustment of the clustering threshold for fine-tuning message grouping.
- **Dimensionality Reduction**: Implements Principal Component Analysis (PCA) to manage and reduce the dimensionality of the high-dimensional vector data efficiently. This technique is essential for maintaining performance and accuracy in similarity searches and clustering by reducing computational complexity while preserving the vectors' essential characteristics.
- **Similar Message Retrieval (`find_similar_messages`)**: Utilizes dimensionality reduction to streamline the retrieval of messages that are contextually similar to a given query vector. This feature is crucial for providing contextually relevant responses and ensuring the system's efficiency in processing and organizing large volumes of data.
- **Persistence (`save`, `load`)**: Provides mechanisms for saving the current state of the vector database to disk and loading an existing database, facilitating continuity across sessions.

## Shell Command Interpretation and Execution (use with caution)

An innovative feature that interprets natural language descriptions of desired system actions and translates them into executable shell commands, enhancing the chat interface's functionality.

### Feature Overview

- **Natural Language Command Interpretation**: Translates user-described actions into executable shell commands, providing an intuitive way to perform system tasks directly from the chat interface.
- **Security and Validation**: Includes measures to ensure the safety and authorization of commands to be executed, maintaining system integrity.
- **Seamless Integration**: The feature is integrated into the chat interface, allowing users to perform tasks without specific command knowledge.

### How to Use

To use this feature, describe the action prefixed with "!shell". For example:
```
!shell delete temporary files in the current directory
```
The system will interpret the action, execute the corresponding shell command, and display the output.
```
C:\windows\system32>format c: /fs:NTFS
The type of the file system is NTFS.

WARNING, ALL DATA ON NON-REMOVABLE DISK
DRIVE C: WILL BE LOST!
Proceed with Format (Y/N) Y

Format is formatting the volume while giving you a cautionary tale.
100% complete.
>_
```

## Server Setup for Shell Command Execution

A Flask-based server setup is required for executing shell commands received from the chat system, acting as an intermediary to perform actions on the server's host system.

### Server Code Overview

A Flask application with an `/execute` endpoint listens for POST requests containing shell commands. It includes authentication via a secret key to prevent unauthorized access.

### Installation and Usage

1. **Dependencies**: Install the required Python packages.
2. **Configuration**: Configure the system with Ollama and set up the Flask server for command execution.
3. **Execution**: Run the Jupyter Notebook to start the interactive chat interface.

### Visualizing Message Clusters

Utilize `vector_db.plot_dendrogram()` to visualize the clustering of messages, offering insights into the conversation's thematic structure.

 ![download (33)](https://github.com/cbrwx/VectorDatabaseAssistedChat/assets/81207429/6fe3b10f-3ac7-42cb-afb1-51b7ebe61bfc)

### Integration with LLMs

Direct interaction with LLMs through Ollama for dynamic response generation, enabling the system to provide contextually aware responses based on the vector database's history and the current conversation context.

## Future Enhancements

- **Encoding Efficiency**: Exploring alternative sentence embedding models for improved performance.
- **LLM Expansion**: Extending interactions to multiple LLMs for a broader range of responses and capabilities.
- **User Interface Enhancements**: Further developing the interactive interface for an even more engaging and intuitive user experience, thru both api and commandline.
- Implement a structured programming task management and integration system using !codetask command, outlining program structure, managing tasks sequentially, and integrating code snippets into a cohesive program with continuous model updates and feedback for error corrections.
- Implement whitelist for commands within shell_server.

.cbrwx
