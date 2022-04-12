from flask import Flask, render_template
from markupsafe import escape
import os

app = Flask(__name__)

@app.route('/')
def show_image(name):
	return render_template('index.html', user_image='samples/image0001.png')