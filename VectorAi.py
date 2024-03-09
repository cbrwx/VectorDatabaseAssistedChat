import json
import requests
from IPython.display import display
import ipywidgets as widgets
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import cdist
import pickle
import warnings
import torch
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
global global_conversation_history
global_conversation_history = ""

class SimpleVectorDatabase:
    def __init__(self, filepath=None, clustering_threshold=1.0):
        self.filepath = filepath
        self.vectors = []
        self.messages = []
        self.types = []
        self.clustering_threshold = clustering_threshold
        self.linkage_matrix = None
        self.cluster_labels = [] 
        if self.filepath:
            self.load()

    def encode_message(self, message, model_encoder=None, chunk_size=256):
        if model_encoder is None:
            model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        words = message.split()
        chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
        vectors = [model_encoder.encode(chunk, show_progress_bar=False, convert_to_tensor=True) for chunk in chunks]
        vector = torch.mean(torch.stack(vectors), dim=0) if vectors else np.zeros(model_encoder.get_sentence_embedding_dimension())
        return vector.cpu().numpy()

    def add_vector(self, vector, message, msg_type):
        self.vectors.append(vector)
        self.messages.append({'message': message, 'type': msg_type})
        self._update_clusters()
        self.save()

    def add_interaction(self, query_vector, query_message, response_vector, response_message):
        self.add_vector(query_vector, query_message, 'query')
        self.add_vector(response_vector, response_message, 'response')
        
    def adjust_clustering_threshold(self, new_threshold):
        self.clustering_threshold = new_threshold
        self._update_clusters()       

    def find_similar_messages(self, vector, n=1):
        if not self.vectors:
            return []

        # Temporarily append the query vector to perform clustering
        temp_vectors = np.vstack([self.vectors, vector])
        temp_linkage_matrix = linkage(temp_vectors, method='ward')
        temp_cluster_labels = fcluster(temp_linkage_matrix, t=self.clustering_threshold, criterion='distance')
        query_cluster = temp_cluster_labels[-1]

        # Filter vectors by the query's cluster
        cluster_indices = [i for i, label in enumerate(self.cluster_labels) if label == query_cluster]
        filtered_vectors = [self.vectors[i] for i in cluster_indices]

        if filtered_vectors:
            distances = cdist([vector], filtered_vectors, metric='cosine').flatten()
            nearest_indices = np.argsort(distances)[:n]
            return [self.messages[cluster_indices[i]] for i in nearest_indices]
        else:
            return []

    def _update_clusters(self):
        if len(self.vectors) > 1:
            self.linkage_matrix = linkage(self.vectors, method='ward')
            self.cluster_labels = fcluster(self.linkage_matrix, t=self.clustering_threshold, criterion='distance')
        else:
            self.cluster_labels = np.zeros(len(self.vectors))

    def save(self):
        with open(self.filepath, 'wb') as f:
            data = {
                'vectors': self.vectors,
                'messages': self.messages,
                'types': self.types,
                'linkage_matrix': self.linkage_matrix,
                'cluster_labels': self.cluster_labels  # Save cluster labels
            }
            pickle.dump(data, f)

    def load(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.messages = data['messages']
                self.types = data.get('types', [])
                self.linkage_matrix = data.get('linkage_matrix', None)
                if self.linkage_matrix is not None:
                    self._update_clusters()
        except (FileNotFoundError, EOFError):
            pass
        
    def plot_dendrogram(self):
        if self.linkage_matrix is not None:
            with plt.style.context('dark_background'):  # Use dark theme
                plt.figure(figsize=(10, 7))                
                dendrogram(self.linkage_matrix, color_threshold=1, above_threshold_color='#add8e6')
                plt.title("Hierarchical Clusterfucking Dendrogram", color='#add8e6')  
                plt.xlabel("Sample index", color="#add8e6")  #
                plt.ylabel("Distance", color="#add8e6")
                plt.xticks(color="#add8e6")  
                plt.yticks(color="#add8e6")
                plt.show()
        else:
            print("Link to the matrix unreachable, you need longer hands.")         

vector_db = SimpleVectorDatabase(filepath='vector_database.pkl')     

def encode_message_to_vector(message, model_encoder=None, chunk_size=256):
    # Ensure chunk_size is in terms of tokens for the model, adjust as needed
    # Split the message into chunk_size-word chunks
    words = message.split()
    chunks = [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    
    # Initialize model_encoder if it's not passed as an argument
    if model_encoder is None:
        model_encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    vectors = []
    for chunk in chunks:
        # Ensure the model does not receive too large of a chunk by checking its token length
        encoded_chunk = model_encoder.encode(chunk, show_progress_bar=False, convert_to_tensor=True)
        vectors.append(encoded_chunk)
    
    # Aggregate the vectors by taking the mean vector if there are multiple chunks
    if vectors:
        vector = torch.mean(torch.stack(vectors), dim=0)
    else:
        # Handle the case where the message might be empty or too short
        vector_dimension = model_encoder.get_sentence_embedding_dimension()
        vector = np.zeros(vector_dimension)
    
    # Convert the tensor back to a numpy array if needed
    if isinstance(vector, torch.Tensor):
        vector = vector.cpu().numpy()
    
    return vector

def chat(user_input, context_messages=[]):
    global global_conversation_history
    
    # Fetch similar context messages from the vector database based on the current user input
    input_vector = encode_message_to_vector(user_input)
    similar_messages = vector_db.find_similar_messages(input_vector, n=5)
    # Assuming 'find_similar_messages' returns a list of message strings
    vector_based_context = "\n".join([f"\033[96mVector Context:\033[0m {msg}" for msg in similar_messages])

    instruction = "\n--- Respond concisely, focusing on the user's current question. Below is the context provided:"
    
    if context_messages:
        recent_context = context_messages[-1]  # last message is the most relevant
        background_context = " ".join(context_messages[:-1]) if len(context_messages) > 1 else ""        
        recent_context_formatted = f"\n\x1b[92m--- Most Recent Context:\x1b[0m\n{recent_context}."
        background_context_formatted = f"\n\x1b[92m--- Background Information Cluster:\x1b[0m\n{background_context}." if background_context else ""
        full_context = f"{instruction} {recent_context_formatted} {background_context_formatted}"
    else:
        full_context = instruction

    # Update the global conversation history with clear labeling for user and model, trim at 5000 chars
    if len(global_conversation_history) > 5000:
        global_conversation_history = global_conversation_history[-5000:]
    global_conversation_history += f"\nUser: {user_input}"

    user_query = f"\n\x1b[1;92mPriority Instruction:\x1b[0m Please address the user's immediate question detailed below with a focused response. Use all relevant contextual information provided as if it were part of your internal knowledge base, understanding that the user does not have visibility into this background information. Your reply should seamlessly reflect this context as if recalling from memory, utilizing it to enhance the clarity and relevance of your answer. Do not reference the context explicitly, but apply it to inform your response effectively. All prior context serves to underpin and guide your understanding in addressing this specific query. Treat the subsequent text after the colon as the actual instructions, which is the core subject of your response: {user_input}"

    # Include both the vector-based context and conversation history in the message sent to the model
    full_message = f"{full_context}\n{vector_based_context}\n\033[92m--- The Conversation History is presented solely for contextual understanding and is not to be considered relevant for the response to the immediate inquiry:\033[0m (up to 5000 chars):\n{global_conversation_history}\n{user_query}"

    print("\nSending the following structured message to the model for context:\n")
    print(full_message)    
    
    print("\n-----------------------------------------\n")
    print("\x1b[1;92m[Updating MainFrame]\x1b[0m\n")

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "odinai",
                "messages": [{"role": "user", "content": full_message}],
                "stream": True,
            },
        )
        response.raise_for_status()

        # Process the response from the model
        output = ""
        for line in response.iter_lines():
            if line:
                body = json.loads(line)
                if "error" in body:
                    raise Exception(body["error"])
                if not body.get("done", False):
                    content = body.get("message", {}).get("content", "")
                    output += content
                else:
                    break

        # Trim before appending to keep the last 2000 chars
        if len(global_conversation_history) + len(f"\nOdinAI: {output}") > 5000:
            global_conversation_history = global_conversation_history[-(5000-len(f"\nOdinAI: {output}")):]
        global_conversation_history += f"\nOdinAI: {output}"

        return {"content": output}
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return {"content": "Error processing your request."}

# initialize little cars
text_area = widgets.Textarea(
    placeholder='Type your message here...',
    description='Input:',
    disabled=False,
    layout=widgets.Layout(width='98%', height='200px')
)
send_button = widgets.Button(
    description='Send',
    disabled=False,
    button_style='',
    tooltip='Send',
    icon='send'
)
output_field = widgets.Output()

# display stales
display(text_area, send_button, output_field)

def on_send_button_clicked(b):
    with output_field:
        output_field.clear_output()  
        user_input = text_area.value
        if not user_input:
            print("Please enter a message.")
            return

        # Directly encode the user input using the updated SimpleVectorDatabase method
        input_vector = vector_db.encode_message(user_input)
        
        # Add the user input and its vector representation to the database
        vector_db.add_vector(input_vector, user_input, 'user')  # Note: 'user' is the message type

        # Retrieve similar messages based on the input vector. This step inherently benefits from hierarchical clustering
        similar_messages = vector_db.find_similar_messages(input_vector, n=5)
        # Assuming 'find_similar_messages' returns a list of dictionaries with 'message' keys
        context_messages = [msg['message'] for msg in similar_messages]

        # Process the input, now including similar context messages for a more informed response
        response = chat(user_input, context_messages)
        print(f"\x1b[1;92mOdinAI:\x1b[0m {response['content']}")

        # Encode the response from the model and add it to the database as a 'model' type message
        if response['content']:
            response_vector = vector_db.encode_message(response['content'])
            vector_db.add_vector(response_vector, response['content'], 'OdinAI')  # 'OdinAI' is the message type

        # Clear the input field after processing
        text_area.value = ''
        vector_db.plot_dendrogram()

# Attach event to the send button#cbrwx
send_button.on_click(on_send_button_clicked)
