from stylegan_generator import StyleGANGenerator

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-256x256.pkl'
generator = StyleGANGenerator(network_pkl)

n_batch = 4000
batch_size = 64
outdir = '/home/robert/data/diploma-thesis/datasets/stylegan3/tpsi_1_256x256'
truncation_psi = 1
z_path = '/home/robert/data/diploma-thesis/datasets/stylegan3/tpsi_1/latents/sample_z.h5'

# generator.generate_images(n_batch, batch_size, outdir, truncation_psi=truncation_psi)
generator.generate_images_from_z(z_path, n_batch, batch_size, outdir, truncation_psi=truncation_psi)
# generator.    qtest_generated_images(outdir, 1, truncation_psi=truncation_psi)