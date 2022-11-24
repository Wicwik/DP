from stylegan_generator import StyleGANGenerator

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl'
generator = StyleGANGenerator(network_pkl)

n_batch = 16000
batch_size = 16
outdir = './data/generated_images/'
truncation_psi = 1

generator.generate_images(n_batch, batch_size, outdir, truncation_psi=truncation_psi)
# generator.test_generated_images(outdir, 1, truncation_psi=0.6)