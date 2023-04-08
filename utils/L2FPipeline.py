from torchvision import transforms
import numpy as np
import cv2

import torch

class L2FPipeline:
    def __init__(self, generator, classifier, tpsi = 1, img_size = (218, 178)) -> None:
        self.device = torch.device('cuda')
        self.generator = generator
        self.classifier = classifier.to(self.device)
        self.tpsi = tpsi
        self.img_size = img_size
    
    def transform(self, z):
        to_tensor = transforms.ToTensor()
        imgs = self.generator.generate_from(z, truncation_psi=self.tpsi)
        # imgs = torch.stack([to_tensor(cv2.resize(img, dsize=self.img_size, interpolation=cv2.INTER_LANCZOS4)).to(self.device) for img in imgs])
        imgs = transforms.Resize(size=self.img_size)(imgs).to(self.device)
        print(imgs.shape)
        
        preds = None
        with torch.no_grad():
            preds = self.classifier(imgs).cpu()
            print(preds)
            return preds, imgs.cpu().numpy()