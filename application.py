from flask import Flask, render_template
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './samples'

@app.route('/')
@app.route('/index')
def show_index():
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image0001.png')
	return render_template('index.html', user_image=full_filename)