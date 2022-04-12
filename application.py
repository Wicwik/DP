from flask import Flask, render_template
from markupsafe import escape
import os

app = Flask(__name__)

PEOPLE_FOLDER = os.path.join('static', 'samples')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

@app.route('/')
@app.route('/index')
def show_index():
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image0001.png')
	return render_template('index.html', user_image=full_filename)