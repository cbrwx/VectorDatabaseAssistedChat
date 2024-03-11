![f323f](https://github.com/cbrwx/VectorDatabaseAssistedChat/assets/81207429/06fe406c-eaf4-4aaf-8061-5461c3639a83)
# OdinAI; a context-Aware Chat System with Vector Database Memory

This repository contains the implementation of a context-aware chat system that integrates a vector database for enhancing the contextual understanding and response quality of a chat interface. This system leverages the power of sentence embeddings, hierarchical clustering, and direct interaction with Large Language Models (LLMs) via Ollama.

## System Overview

### Components

- **Vector Database (`SimpleVectorDatabase`)**: A core component designed to encode chat messages into high-dimensional vectors and store them for contextual retrieval.
- **Context Extraction**: Mechanism to enrich messages with contextual information before encoding, utilizing external LLMs for context identification.
- **LLM Communication**: Direct API calls to LLMs, facilitated by Ollama, for generating responses based on the current and retrieved context.
- **Interactive Interface**: Utilization of IPython widgets for creating an interactive chat environment within Jupyter Notebooks.

### Key Features

- **Message Encoding**: Converts textual messages into vector representations using `SentenceTransformer`, enhancing semantic comparison capabilities.
- **Hierarchical Clustering**: Organizes encoded messages into a hierarchical structure, enabling efficient retrieval of contextually similar messages.
- **Direct LLM Interaction**: Employs Ollama for streamlined communication with LLMs, bypassing the need for intermediary services.
- **Dynamic Context Understanding**: Continuously updates the vector database with new interactions, improving the system's contextual awareness over time.

## Technical Description

### Vector Database (`SimpleVectorDatabase`)

- **Initialization (`__init__`)**: Sets up the vector database with optional filepath for persistence and a threshold for clustering. Initializes internal structures for vectors, messages, types, contexts, and clustering data. Automatically loads existing database if filepath is provided.
- **Context Extraction (`extract_context`)**: For each input message, this function contacts an LLM via Ollama to extract relevant context, enhancing the message with additional semantic information before encoding. This process involves sending the message to a pre-configured LLM endpoint and parsing the response to capture the message's context.
- **Message Encoding (`encode_message`)**: Utilizes `SentenceTransformer` to convert messages into dense vector representations. This function handles context inclusion, combining the extracted context with the original message for a comprehensive encoding. Messages are chunked to accommodate model limitations, with each chunk encoded separately before aggregation to form the final message vector.
- **Vector Addition (`add_vector`)**: Adds a new vector to the database, ensuring no duplicates. It optionally extracts context if not provided, stores the vector along with its metadata (message, type, context), and updates the message clusters.
- **Interaction Recording (`add_interaction`)**: Facilitates the recording of a query-response interaction by encoding and adding both components to the database. This function is crucial for building a rich historical context within the system.
- **Clustering Threshold Adjustment (`adjust_clustering_threshold`)**: Allows dynamic adjustment of the clustering threshold, enabling fine-tuning of how messages are grouped based on similarity.
- **Similar Message Retrieval (`find_similar_messages`)**: Implements a key feature of the system, retrieving messages that are contextually similar to a given query vector. This process involves temporary augmentation of the database with the query vector, re-clustering, and identifying messages within the same cluster as the query for similarity comparison.
- **Cluster Update (`_update_clusters`)**: A private method that recalculates message clusters based on the current threshold, ensuring the database reflects the latest organizational structure.
- **Persistence (`save`, `load`)**: Provides mechanisms for saving the current state of the vector database to disk and loading an existing database, facilitating continuity across sessions.

## Shell Command Interpretation and Execution

This update introduces a dynamic shell command interpretation and execution feature within our chat interface. This innovative functionality allows users to describe their desired system actions in natural language, which the system then interprets and translates into executable shell commands.

### Feature Overview

- **Natural Language Command Interpretation:** Users can describe what they want to accomplish in natural language, and the system interprets these descriptions to generate and execute the corresponding shell command(s).
- **Security and Validation:** Implements robust validation and security measures to ensure that only safe and authorized commands are interpreted and executed.
- **Seamless User Experience:** The feature is seamlessly integrated into the chat interface, providing users with an intuitive way to perform system tasks directly from the chat without needing specific command knowledge.

### How to Use

Simply describe the action you wish to perform in the chat, prefixed with a specific trigger phrase. For example:
```
!shell delete the temporary files in the current directory(1*)

or

!shell write me a poem and put it in a file called turtles.txt, make it deep and without chelonitoxism.
```
The system will interpret the request, for example identify the most likely corresponding shell command (e.g., `rm -rf /path/to/temp/*`)(1*), and execute it after performing necessary security checks and yet to be built whitelisted command filter.

## Server Setup for Shell Command Execution

To facilitate the execution of shell commands through the chat interface, a server using the Flask framework is required. This server acts as an intermediary, receiving commands from the chat system, executing them on the server's host system, and returning the results.

### Server Code Overview

The provided server code snippet defines a simple Flask application with a single endpoint `/execute`, which listens for POST requests containing shell commands to execute. Authentication is handled through a secret key to prevent unauthorized access.

```python
from flask import Flask, request, jsonify
import subprocess

SECRET_KEY = "your_secret_key_here"

app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.json
    command = data.get('command')
    provided_secret_key = data.get('secret_key')
    
    if provided_secret_key != SECRET_KEY:
        return jsonify({'error': 'Unauthorized access attempt. The secret key is invalid.'}), 401

    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return jsonify({'output': output}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.output}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)
```

### Message Processing Workflow

1. **User Input**: Received through a widget-based interface.
2. **Context and Encoding**: Each message is contextually enriched and encoded into a vector.
3. **Database Update**: The vector, along with its metadata, is stored in the database.
4. **Contextual Retrieval**: For each new query, the system retrieves relevant past messages based on vector similarity.
5. **Response Generation**: Combining the new query with retrieved context, a comprehensive request is sent to an LLM via Ollama, which generates a contextually aware response.

### Installation and Usage

1. **Dependencies**: Install necessary Python packages listed in `requirements.txt`.
2. **Configuration**: Set up Ollama with the desired LLM for response generation.
3. **Execution**: Run the provided Jupyter Notebook to interact with the chat system.

### Visualizing Message Clusters

- **Dendrogram Visualization**: Call `vector_db.plot_dendrogram()` to visualize the clustering of messages, providing insights into the semantic organization of the database.

![download (33)](https://github.com/cbrwx/VectorDatabaseAssistedChat/assets/81207429/6fe3b10f-3ac7-42cb-afb1-51b7ebe61bfc)

### Integration with Ollama

- **Purpose**: Enables direct interaction with configured LLMs for dynamic response generation.
- **Mechanism**: Constructs and sends JSON payloads containing the user's query and relevant context to the LLM, processing the received responses for display.

## Future Enhancements

- **Optimization**: Explore alternative sentence embedding models for improved encoding efficiency.
- **LLM Expansion**: Extend the system to interact with multiple LLMs for diverse response generation and task generation/completion(for large projects like books(done), games, etc).
- **User Experience**: At some point; enhancements to the interactive interface for a more engaging and intuitive user interaction.

.cbrwx
