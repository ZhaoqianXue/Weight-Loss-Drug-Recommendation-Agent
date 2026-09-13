import os
from flask import Flask, render_template, request, jsonify, send_from_directory
# Make sure your chatbot.py is in the same directory or accessible in PYTHONPATH


app = Flask(__name__)

# --- Configuration ---
# It's highly recommended to use environment variables for sensitive data like API keys
# For example, you would set OPENAI_API_KEY in your PythonAnywhere environment
# OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
# NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687") # Default if not set

# --- Initialize Chatbot ---
# This can take time, so it's done once when the app starts.
# Ensure your chatbot.py handles potential errors during initialization gracefully.
# Also, make sure paths within chatbot.py (to CSV, FAISS DBs) are correct
# relative to where this Flask app is run, or are absolute paths.
chatbot_instance = None
try:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured; static graph remains available")
    from chatbot import MedicalChatBot
    print("Initializing MedicalChatBot...")
    # You might need to pass configurations like API keys to your chatbot constructor
    # if you modify chatbot.py to accept them instead of using hardcoded values.
    chatbot_instance = MedicalChatBot(verbose=True) # Set verbose to False in production
    print("MedicalChatBot initialized successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to initialize MedicalChatBot: {e}")
    chatbot_instance = None # Ensure it's None if initialization fails

# --- Routes ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    # Assumes knowledge_graph.html is in a 'templates' folder
    # in the same directory as flask_app.py
    return render_template('knowledge_graph.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the frontend."""
    if not chatbot_instance:
        return jsonify({"error": "Chatbot is not available due to an initialization error."}), 500

    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    try:
        # Get response from your chatbot logic
        bot_response = chatbot_instance.get_response(user_input)
        return jsonify({"reply": bot_response})
    except Exception as e:
        print(f"Error during chatbot query: {e}")
        # Log the full error for debugging: import traceback; traceback.print_exc();
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serves static files (like the CSV for the frontend)."""
    # Assumes your CSV is in a 'static' folder.
    # The frontend JavaScript `Papa.parse('standardized_reviews_all.csv', ...)`
    # should be changed to `Papa.parse('/static/standardized_reviews_all.csv', ...)`
    return send_from_directory('static', filename)

# --- For PythonAnywhere WSGI setup ---
# PythonAnywhere will look for a Flask app instance named 'app'.
# If your main script is named flask_app.py, the WSGI file on PythonAnywhere
# would typically be configured to import 'app' from 'flask_app'.

if __name__ == '__main__':
    # This is for local development.
    # PythonAnywhere uses a WSGI server, not Flask's built-in development server.
    app.run(debug=True, port=5001)
