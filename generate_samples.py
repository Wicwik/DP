from stylegan_generator import StyleGANGenerator

network_pkl = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl'
generator = StyleGANGenerator(network_pkl)

n_batch = 32000
batch_size = 16
outdir = '/home/robert/data/diploma-thesis/datasets/stylegan2/tpsi_1/'
truncation_psi = 1
z_path = '/home/robert/data/diploma-thesis/datasets/stylegan2/tpsi_1/latents/sample_z.h5'

generator.generate_images(n_batch, batch_size, outdir, truncation_psi=truncation_psi)
# generator.generate_images_from_z(z_path, n_batch, batch_size, outdir, truncation_psi=truncation_psi)
# generator.    qtest_generated_images(outdir, 1, truncation_psi=truncation_psi)