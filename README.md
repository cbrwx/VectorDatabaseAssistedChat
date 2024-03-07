
![sdfsdfsdfds](https://github.com/cbrwx/VectorDatabaseAssistedChat/assets/81207429/7dbb6018-4d36-4507-b032-87dbfbf63a0b)

# Vector Database-Assisted Chat Interface 
Outlines the implementation details and usage instructions for a chat interface that utilizes a vector database for storing and retrieving conversation history to enhance the contextual relevance of responses. The project integrates local language model management via Ollama and leverages Jupyter Notebook for its interactive interface.

# Project Overview
The core of this project is to enhance chatbot interactions by incorporating a context-aware mechanism. This is achieved by encoding textual messages into vectors and storing them in a custom-built vector database. The database facilitates the retrieval of contextually similar past interactions, which inform the generation of relevant responses to new queries.

# Components
## Vector Database
A custom vector database is used in this project; it stores vectors and their corresponding messages, allowing for the addition of new interactions and the retrieval of similar past messages based on vector similarity.

## Encoding Messages
The SentenceTransformer library is utilized for converting textual messages into vector representations. This conversion is crucial for comparing semantic similarities between messages, enabling the system to identify and retrieve relevant past interactions.

## Interaction with LLMs
The interface communicates with external Large Language Models (LLMs) via Ollama, a local manager that facilitates API interactions with these models. Ollama serves as the intermediary, managing the request and response flow between the user interface and the LLM.

## User Interface
The project employs IPython widgets within a Jupyter Notebook to create an interactive chat interface. This setup allows users to input queries, to which the system responds with contextually informed answers.

# How It Works
## Initialization
Upon starting, the vector database attempts to load any pre-existing data if a file path is specified, allowing for continuity across sessions. The SentenceTransformer model is initialized for encoding purposes.

## User Interaction
Users interact with the chatbot through a text input widget. Upon submission of a query:

- Encoding: The input text is encoded into a vector using the specified transformer model.
- Context Retrieval: The system queries the vector database for messages with similar vectors, retrieving a set of past interactions deemed contextually relevant.
- Response Generation: The current query, along with the retrieved context, is sent to an external LLM via Ollama. The LLM generates a response based on this composite input.
- Update Database: Both the query and the generated response are encoded into vectors and stored in the database for future reference.
# Communication with LLM
The system constructs a JSON payload containing the user's query and the relevant context, which is sent to the LLM through Ollama. The response from the LLM is processed and displayed to the user.

# Setup and Configuration
## Prerequisites
A functional Python environment.
Jupyter Notebook for the interactive interface.
Ollama configured to manage the chosen LLM (e.g., dolphin-mistral, mistral, or mixtral).
## Installation
Ensure all necessary Python packages are installed, including numpy, scipy, sentence_transformers, ipywidgets, and requests. Ollama should be set up according to its documentation, with the LLM of choice configured within its settings.

## Running the Interface
Launch the Jupyter Notebook and execute the cells to initialize the components. Interact with the chatbot using the provided text area widget.

# Technical Details
## Vector Similarity
Cosine similarity is used to compare vectors, identifying which stored messages are most relevant to the current query. This method ensures that the retrieval of past interactions is based on semantic similarity rather than surface-level text matching.

## Database Management
The vector database supports dynamic updates, allowing new interactions to be seamlessly integrated into the dataset. It is designed for efficient querying, ensuring that response generation is both timely and relevant.

## Ollama Integration
Ollama's role as a local LLM manager is critical for abstracting away the complexities of API communication with the LLM. This simplifies the process of sending requests and processing responses, making the overall system more robust and easier to maintain. (https://ollama.ai/)

.cbrwx
