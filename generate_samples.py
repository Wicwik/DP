from stylegan_generator import StyleGANGenerator


network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl'
generator = StyleGANGenerator(network_pkl)

n_samples = 10
outdir = 'static/samples'

generator.generate_images(n_samples, outdir, truncation_psi=0.6)