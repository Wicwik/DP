from flask import Flask, render_template, send_file, request
from markupsafe import escape
import os
import io

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

def serve_pil_image(pil_img):
	img_io = io.BytesIO()
	pil_img.save(img_io, 'PNG', quality=70)
	img_io.seek(0)
	return send_file(img_io, mimetype='image/png')

@app.route('/generate/json/new', methods=['POST'])
def serve_img_json():
	request_data = request.get_json()

	truncation_psi = 1

	if request_data:
		if 'truncation_psi' in request_data:
			truncation_psi = request_data['truncation_psi']

	img = generator.generate_one(truncation_psi)
	return serve_pil_image(img)

@app.route('/generate', methods=['GET', 'POST'])
def serve_img_form():
	name = 'form_image.png'
	
	os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

	if request.method == 'POST':
		if request.form['action'] == 'Generate':
			z = generator.get_random_vectors(1)
			truncation_psi = request.form.get('truncation_psi')

			full_filename = os.path.join(app.config['UPLOAD_FOLDER'], f'{escape(name)}')
			if truncation_psi == '':
				generator.generate_from(full_filename, z)
			else:
				generator.generate_from(full_filename, z, truncation_psi=float(truncation_psi))

			print(full_filename)
			return render_template('form.html', user_image=full_filename)

	return render_template('form.html')

	

