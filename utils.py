from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

def get_pinned_out(model):
    pinned_out = []

    if isinstance(model, ResNet):
        pinned_out = ['conv1']
        
        for name, module in model.named_modules():
            if isinstance(module, BasicBlock):
                pinned_out.append(f'{name}.conv2')
                if module.downsample is not None:
                    pinned_out.append(f'{name}.downsample.0')
            
            if isinstance(module, Bottleneck):
                pinned_out.append(f'{name}.conv3')
                if module.downsample is not None:
                    pinned_out.append(f'{name}.downsample.0')

    return pinned_out