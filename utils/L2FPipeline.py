class L2FPipeline:
    def __init__(self, generator, classifier, tpsi = 1) -> None:
        self.generator = generator
        self.classifier = classifier
        self.tpsi = tpsi

    def transform(self, z):
        imgs = self.generator.generate_torch(z, truncation_psi=self.tpsi)
        print(imgs.shape)
        print(imgs)