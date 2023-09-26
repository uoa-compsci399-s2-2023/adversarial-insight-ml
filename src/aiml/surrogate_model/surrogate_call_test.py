
from create_surrogate_model import create_surrogate_model
import timm


if __name__ =="__main__":
    model_to_test = timm.create_model("resnet18_cifar10", pretrained=True)
    model = create_surrogate_model(model_to_test)
    




