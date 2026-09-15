"""Flask adapter; the research chatbot starts only on the first /chat request."""
import os
from flask import Flask, jsonify, request, send_from_directory
from weightloss.settings import get_settings

def create_app():
    directory = get_settings().path('web_build')
    app = Flask(__name__, static_folder=str(directory/'static'), static_url_path='/static')
    bot = None

    @app.get('/')
    def home():
        return send_from_directory(directory/'templates', 'knowledge_graph.html')

    @app.post('/chat')
    def chat():
        nonlocal bot
        message = (request.get_json(silent=True) or {}).get('message')
        if not isinstance(message, str) or not message.strip():
            return jsonify(error='No message provided'), 400
        if not os.environ.get('OPENAI_API_KEY'):
            return jsonify(error='Chatbot is not configured; the static assistant remains available.'), 503
        try:
            if bot is None:
                from weightloss.retrieval.chatbot import MedicalChatBot
                bot = MedicalChatBot(verbose=False)
            return jsonify(reply=bot.get_response(message))
        except Exception:
            app.logger.exception('Research chatbot request failed')
            return jsonify(error='The research backend could not complete this request.'), 503

    return app

app = create_app()
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001)
