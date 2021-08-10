import torch 
def get_model(name):
    if name == 'resnet':
        from .basic_embedding import ResNetFeats
        return ResNetFeats
    elif name == 'vgg':
        from .basic_embedding import VGGFeats
        return VGGFeats
    else:
        raise NotImplementedError
        
