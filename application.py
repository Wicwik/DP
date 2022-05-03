from flask import Flask, render_template, send_file
from markupsafe import escape
import os

from stylegan_generator import StyleGANGenerator

app = Flask(__name__)

PEOPLE_FOLDER = os.path.join('static', 'samples')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl'
generator = StyleGANGenerator(network_pkl)

@app.route('/')
@app.route('/index')
def show_index():
	return 'API for diploma thesis'

@app.route('/<name>')
def show_image(name):
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'{escape(name)}')
	return render_template('index.html', user_image=full_filename)

@app.route('/img/<num>')
def get_image(num):
	name = f'image{int(num):04d}.png'
	full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'{escape(name)}')
	return send_file(full_filename, mimetype='image/png')

@app.route('/generate/<n_samples>')
def generate_images(n_samples):
	outdir = 'static/samples'
	generator.generate_images(int(n_samples), outdir, truncation_psi=0.6)
	return 'Succesfully generated {} images'.format(n_samples)
