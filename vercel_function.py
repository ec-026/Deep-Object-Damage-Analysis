# vercel_function.py
from flask import Flask
from webapp import app  # Import your Flask app from webapp.py


def handler(request, response):
    # You can handle requests here using your Flask app
    return app(request.environ, response)
