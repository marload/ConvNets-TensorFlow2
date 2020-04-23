def choose_nets(nets_name, num_classes=100):
    nets_name = nets_name.lower()
    if nets_name == 'vgg11':
        from models.VGG import VGG11
        return VGG11(num_classes)
    if nets_name == 'vgg13':
        from models.VGG import VGG13
        return VGG13(num_classes)
    if nets_name == 'VGG16':
        from models.VGG import VGG16
        return VGG16(num_classes)
    if nets_name == 'vgg19':
        from models.VGG import VGG19
        return VGG19(num_classes)
    if nets_name == 'resnet18':
        from models.ResNet import ResNet18
        return ResNet18(num_classes)
    if nets_name == 'resnet34':
        from models.ResNet import ResNet34
        return ResNet34(num_classes)
    if nets_name == 'resnet50':
        from models.ResNet import ResNet50
        return ResNet50(num_classes)
    if nets_name == 'resnet101':
        from models.ResNet import ResNet101
        return ResNet101(num_classes)
    if nets_name == 'resnet152':
        from models.ResNet import ResNet152
        return ResNet152(num_classes)
    if nets_name == 'googlenet':
        from models.GoogLeNet import GoogLeNet
        return GoogLeNet(num_classes)
    if nets_name == 'mobilenet':
        from models.MobileNet import MobileNet
        return MobileNet(num_classes)
    if nets_name == 'mobilenetv2':
        from models.MobileNetV2 import MobileNetV2
        return MobileNetV2(num_classes)
    raise NotImplementedError
