import json
import requests
from IPython.display import display
import ipywidgets as widgets
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pickle
import warnings
import torch
from transformers import LongformerModel, LongformerTokenizer

# Suppress specific deprecation warnings
warnings.filterwarnings("ignore", message="torch.utils._pytree._register_pytree_node is deprecated")
global global_conversation_history
global_conversation_history = ""

class SimpleVectorDatabase:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.vectors = None
        self.messages = []
        if self.filepath:  # If a filepath is provided, attempt to load existing data
            self.load()

    def add_vector(self, vector, message, msg_type):
        vector = np.array(vector)  # Ensure the input vector is a NumPy array
        if self.vectors is None:
            self.vectors = np.array([vector])
        else:
            # Ensure that self.vectors is a numpy array before trying to stack
            if not isinstance(self.vectors, np.ndarray):
                self.vectors = np.array(self.vectors)
            self.vectors = np.vstack([self.vectors, vector])
        self.messages.append(message)
        self.types.append(msg_type)  # Store the type of the message
        self.save()
        
    def add_interaction(self, query_vector, query_message, response_vector, response_message):
        # Adds both the query and the response to the database
        if self.vectors is None:
            self.vectors = np.array([query_vector, response_vector])
        else:
            self.vectors = np.vstack([self.vectors, query_vector, response_vector])
        interaction = {'query': {'vector': query_vector, 'message': query_message},
                       'response': {'vector': response_vector, 'message': response_message}}
        self.messages.append(interaction)
        self.save()        

    def find_similar_messages(self, vector, n=1, relevance_threshold=0.5):
        if self.vectors is None or len(self.messages) == 0:
            return []
        similarities = [1 - cosine(vector, stored_vector) for stored_vector in self.vectors]
        relevant_message_indices = [i for i, similarity in enumerate(similarities) if similarity >= relevance_threshold]     
        sorted_relevant_indices = sorted(relevant_message_indices, key=lambda i: similarities[i], reverse=True)[:n]
  
        return [self.messages[i] for i in sorted_relevant_indices]
    
    def save(self):
        if not self.filepath:
            return  # Do nothing if no filepath is set
        with open(self.filepath, 'wb') as f:
            pickle.dump({'vectors': self.vectors, 'messages': self.messages, 'types': self.types}, f)

    def load(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.messages = data['messages']
                self.types = data.get('types', [])  # Ensure backward compatibility
        except (FileNotFoundError, EOFError):
            self.vectors = None
            self.messages = []
            self.types = []
        
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
        background_context_formatted = f"\n\x1b[92m--- Background Information:\x1b[0m\n{background_context}." if background_context else ""
        full_context = f"{instruction} {recent_context_formatted} {background_context_formatted}"
    else:
        full_context = instruction

    # Update the global conversation history with clear labeling for user and model, trim at 5000 chars
    if len(global_conversation_history) > 5000:
        global_conversation_history = global_conversation_history[-5000:]
    global_conversation_history += f"\nUser: {user_input}"

    user_query = f"\n\x1b[1;92mPriority Instruction:\x1b[0m Respond directly to the user's current query below. It is imperative that this query is the main focus of your response. While formulating your answer, please integrate and consider all available contextual information provided. This query is the central point of your response, remember the user cannot see all your context, so reply to the user like its from your memory not from the context i give you in the beginning of the message Remeber the user cannot see your context so no need to explain it but rather use your context as data! And all other context should support in understanding and addressing this specific inquiry, So again consider everything before this as context and memory, this is your MAIN TASK: {user_input}"

    # Include both the vector-based context and conversation history in the message sent to the model
    full_message = f"{full_context}\n{vector_based_context}\n\033[92m--- Conversation History:\033[0m (up to 5000 chars):\n{global_conversation_history}\n{user_query}"

    print("\nSending the following structured message to the model for context:\n")
    print(full_message)    
    
    print("\n-----------------------------------------\n")
    print("\x1b[1;92m[Updating MainFrame]\x1b[0m\n")    

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
#                 "model": "dolphin-mistral",
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

        # Trim before appending to keep the last 5000 chars as conversation history
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

# display stales and join discord
display(text_area, send_button, output_field)

def on_send_button_clicked(b):
    with output_field:
        output_field.clear_output()  # Clear the previous outputs
        user_input = text_area.value
        if not user_input:
            print("Please enter a message.")
            return

        #print(f"User: {user_input}")
        input_vector = encode_message_to_vector(user_input)
        vector_db.add_vector(input_vector, user_input, 'user')  # Add user input as 'user'

        # Retrieve similar messages as context and process the input
        similar_messages = vector_db.find_similar_messages(input_vector, n=5)
        context_messages = [msg for msg in similar_messages]
        response = chat(user_input, context_messages)
        print(f"\x1b[1;92mOdinAI:\x1b[0m {response['content']}")

        # After receiving the response from the model, encode it and add to the database as a 'model' type message
        if response['content']:
            response_vector = encode_message_to_vector(response['content'])
            vector_db.add_vector(response_vector, response['content'], 'model')  # Add model response as 'model'

        text_area.value = ''

# Attach event to the send button
send_button.on_click(on_send_button_clicked)
