from flask import Flask, render_template
from markupsafe import escape
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './samples'

@app.route('/<name>')
def show_image(name):
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], '{escape(name)}')
	return render_template('index.html', user_image=full_filename)