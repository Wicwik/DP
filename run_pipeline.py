from utils.L2FPipeline import L2FPipeline
from stylegan_generator import StyleGANGenerator
from models.MultilabelResnetClassifier import MultilabelResnetClassifier

network_pkl = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-ffhqu-1024x1024.pkl'

generator = StyleGANGenerator(network_pkl)
classifier = MultilabelResnetClassifier()
pipeline = L2FPipeline(generator = generator, classifier = classifier)
pipeline.transform()