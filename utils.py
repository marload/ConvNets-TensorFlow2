def choose_nets(nets_name, num_classes=100):
    nets_name = nets_name.lower()
    if nets_name == 'vgg11':
        from models.vgg import vgg11
        return vgg11(num_classes)
    if nets_name == 'vgg13':
        from models.vgg import vgg13
        return vgg13(num_classes)
    if nets_name == 'vgg16':
        from models.vgg import vgg16
        return vgg16(num_classes)
    if nets_name == 'vgg19':
        from models.vgg import vgg19
        return vgg19(num_classes)
    if nets_name == 'resnet18':
        from models.resnet import resnet18
        return resnet18(num_classes)
    if nets_name == 'resnet34':
        from models.resnet import resnet34
        return resnet34(num_classes)
    if nets_name == 'resnet50':
        from models.resnet import resnet50
        return resnet50(num_classes)
    if nets_name == 'resnet101':
        from models.resnet import resnet101
        return resnet101(num_classes)
    if nets_name == 'resnet152':
        from models.resnet import resnet152
        return resnet152(num_classes)
    if nets_name == 'googlenet':
        from models.googlenet import googlenet
        return googlenet(num_classes)
    if nets_name == 'preactresnet18':
        from models.preactresnet import preactresnet18
        return preactresnet18(num_classes)
    if nets_name == 'preactresnet34':
        from models.preactresnet import preactresnet34
        return preactresnet34(num_classes)
    if nets_name == 'preactresnet50':
        from models.preactresnet import preactresnet50
        return preactresnet50(num_classes)
    if nets_name == 'preactresnet101':
        from models.preactresnet import preactresnet101
        return preactresnet101(num_classes)
    if nets_name == 'preactresnet152':
        from models.preactresnet import preactresnet152
        return preactresnet152(num_classes)
    if nets_name == 'mobilenet':
        from models.mobilenet import mobilenet
        return mobilenet(num_classes)
    if nets_name == 'mobilenetv2':
        from models.mobilenetv2 import mobilenetv2
        return mobilenetv2(num_classes)
    raise NotImplementedError
