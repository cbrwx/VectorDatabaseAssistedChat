# servercode for shell commands
from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/execute', methods=['POST'])
def execute_command():
    data = request.json
    command = data['command']

    try:
        output = subprocess.check_output(command, shell=True, text=True, stderr=subprocess.STDOUT)
        return jsonify({'output': output}), 200
    except subprocess.CalledProcessError as e:
        return jsonify({'error': e.output}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8000)
