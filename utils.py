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
    if nets_name == 'inceptionv3':
        from models.InceptionV3 import inceptionv3
        return inceptionv3(num_classes)
    if nets_name == 'mobilenet':
        from models.MobileNet import mobilenet
        return mobilenet(num_classes)
    if nets_name == 'mobilenetv2':
        from models.MobileNetV2 import mobilenetv2
        return mobilenetv2(num_classes)
    if nets_name == 'seresnet18':
        from models.SEResNet import seresnet18
        return seresnet18(num_classes)
    if nets_name == 'seresnet34':
        from models.SEResNet import seresnet34
        return seresnet34(num_classes)
    if nets_name == 'seresnet50':
        from models.SEResNet import seresnet50
        return seresnet50(num_classes)
    if nets_name == 'seresnet101':
        from models.SEResNet import seresnet101
        return seresnet101(num_classes)
    if nets_name == 'seresnet152':
        from models.SEResNet import seresnet152
        return seresnet152(num_classes)
    if nets_name == 'densenet121':
        from models.DenseNet import densenet121
        return densenet121(num_classes)
    if nets_name == 'densenet169':
        from models.DenseNet import densenet169
        return densenet169(num_classes)
    if nets_name == 'densenet201':
        from models.DenseNet import densenet201
        return densenet201(num_classes)
    if nets_name == 'densenet121':
        from models.DenseNet import densenet161
        return densenet161(num_classes)
    if nets_name == 'squeezenet':
        from models.SqueezeNet import squeezenet
        return squeezenet(num_classes)
    if nets_name == 'inceptionv4':
        from models.InceptionV4 import inceptionv4
        return inceptionv4(num_classes)
    if nets_name == 'inception-resnet-v2':
        from models.InceptionV4 import inception_resnet_v2
        return inception_resnet_v2(num_classes)
    raise NotImplementedError
