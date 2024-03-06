import json
import requests
from IPython.display import display
import ipywidgets as widgets
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import pickle

class SimpleVectorDatabase:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.vectors = None
        self.messages = []
        if self.filepath:  # if a filepath is provided, attempt to load existing data
            self.load()

    def add_vector(self, vector, message):
        if self.vectors is None:
            self.vectors = np.array([vector])
        else:
            # make sure self.vectors is a numpy array before trying to stack
            if not isinstance(self.vectors, np.ndarray):
                self.vectors = np.array(self.vectors)
            self.vectors = np.vstack([self.vectors, vector])
        self.messages.append(message)
        self.save()  # save the database state after adding a new vector/message
        
    def add_interaction(self, query_vector, query_message, response_vector, response_message):
        # adds both the query and the response to the database
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
            pickle.dump({'vectors': self.vectors, 'messages': self.messages}, f)

    def load(self):
        try:
            with open(self.filepath, 'rb') as f:
                data = pickle.load(f)
                self.vectors = data['vectors']
                self.messages = data['messages']
        except (FileNotFoundError, EOFError):
            pass  
        
vector_db = SimpleVectorDatabase(filepath='vector_database.pkl')        

# initialize SentenceTransformer model for encoding messages to vectors
model_encoder = SentenceTransformer('all-MiniLM-L6-v2')

def encode_message_to_vector(message):
    #print("\nEncoding message:", message)
    return model_encoder.encode(message, show_progress_bar=True)

def chat(user_input, context_messages=[]):
    instruction = "Respond concisely, focusing on the user's current question. Below is the context provided:"
    if context_messages:
        recent_context = context_messages[-1]  # last message is the most relevant
        background_context = " ".join(context_messages[:-1]) if len(context_messages) > 1 else ""

        recent_context_formatted = f"Most Recent Context: {recent_context}."
        background_context_formatted = f"Background Information: {background_context}." if background_context else ""
        
        full_context = f"{instruction} {recent_context_formatted} {background_context_formatted}"
    else:
        full_context = instruction   
    user_query = f"User's Current Query: {user_input}"    
    full_message = f"{full_context} {user_query}"

    # Uncomment below for debugging pleasures
    # print("\nSending the following structured message to the model for context:\n")
    # print(full_message)
    # print("\n-----------------------------------------\n")

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

        # process the response from the model
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

        return {"content": output}
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return {"content": "Error processing your request."}

# initialize little cars
text_area = widgets.Textarea(
    placeholder='Type your message here...',
    description='Input:',
    disabled=False,
    layout=widgets.Layout(width='100%', height='200px')
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
        output_field.clear_output()  # Clear the previous outputs
        user_input = text_area.value
        if not user_input:
            print("Please enter a message.")
            return

        #print(f"User: {user_input}")
        input_vector = encode_message_to_vector(user_input)
        # retrieve similar messages as context
        similar_messages = vector_db.find_similar_messages(input_vector, n=5)
        # extract just the message content for context
        context_messages = [msg for msg in similar_messages]
        # call the chat function with user input and context
        response = chat(user_input, context_messages)
        print(f"CBot: {response['content']}")
        # store the user input and bot response in the vector database
        vector_db.add_vector(input_vector, user_input)
        if response['content']:
            response_vector = encode_message_to_vector(response['content'])
            vector_db.add_vector(response_vector, response['content'])
        text_area.value = ''  
      
# Attach event to the send button
send_button.on_click(on_send_button_clicked)
