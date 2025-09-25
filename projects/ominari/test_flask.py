#!/usr/bin/env python3
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return '<h1>Test Flask App Working!</h1>'

if __name__ == '__main__':
    print("Starting test Flask app on port 5555...")
    app.run(host='127.0.0.1', port=5555, debug=True)