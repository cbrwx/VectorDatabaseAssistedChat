from flask import Flask, request, jsonify
import subprocess

# Define a secret key for authentication
SECRET_KEY = "cbrwx"

app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.json
    
    # Extract the command and the provided secret key from the request
    command = data.get('command')
    provided_secret_key = data.get('secret_key')
    
    # Check if the provided secret key matches the expected secret key
    if provided_secret_key != SECRET_KEY:
        return jsonify({'error': 'Unauthorized access attempt. The secret key is invalid.'}), 401

    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return jsonify({'output': output}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.output}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)
