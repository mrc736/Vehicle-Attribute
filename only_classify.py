from dataset import color_attrs, direction_attrs, type_attrs
import torch
import torchvision
import cv2
import PIL
from PIL import Image
import numpy as np

use_cuda=False
device = torch.device(
    'cuda: 0' if torch.cuda.is_available() and use_cuda else 'cpu')
local_model_path = './checkpoints/epoch_39.pth'
local_car_cfg_path = './car.cfg'
local_car_det_weights_path = './car_detect.weights'


class Cls_Net(torch.nn.Module):
    """
    vehicle multilabel classification model
    """

    def __init__(self, num_cls, input_size):
        """
        network definition
        :param is_freeze:
        """
        torch.nn.Module.__init__(self)

        # output channels
        self._num_cls = num_cls

        # input image size
        self.input_size = input_size

        # delete original FC and add custom FC
        self.features = torchvision.models.resnet18(pretrained=True)
        del self.features.fc
        # print('feature extractor:\n', self.features)

        self.features = torch.nn.Sequential(
            *list(self.features.children()))

        self.fc = torch.nn.Linear(512 ** 2, num_cls)  # 输出类别数
        # print('=> fc layer:\n', self.fc)

    def forward(self, X):
        """
        :param X:
        :return:
        """
        N = X.size()[0]

        X = self.features(X)  # extract features

        X = X.view(N, 512, 1 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (1 ** 2)  # Bi-linear CNN

        X = X.view(N, 512 ** 2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)
        assert X.size() == (N, self._num_cls)
        return X


class Car_Classifier(object):
    """
    vehicle detection model mabager
    """

    def __init__(self,
                 num_cls,
                 model_path=local_model_path):
        """
        load model and initialize
        """

        # define model and load weights
        self.net = Cls_Net(num_cls=num_cls, input_size=224).to(device)
        # self.net = torch.nn.DataParallel(Net(num_cls=20, input_size=224),
        #                                  device_ids=[0]).to(device)
        self.net.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
        print('=> vehicle classifier loaded from %s' % model_path)

        # set model to eval mode
        self.net.eval()

        # test data transforms
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=224),
            torchvision.transforms.CenterCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])

        # split each label
        self.color_attrs = color_attrs
        print('=> color_attrs:\n', self.color_attrs)

        self.direction_attrs = direction_attrs
        print('=> direction attrs:\n', self.direction_attrs)

        self.type_attrs = type_attrs
        print('=> type_attrs:\n', self.type_attrs)

    def get_predict(self, output):
        """
        get prediction from output
        """
        # get each label's prediction from output
        output = output.cpu()  # fetch data from gpu
        pred_color = output[:, :9]
        pred_direction = output[:, 9:11]
        pred_type = output[:, 11:]

        color_idx = pred_color.max(1, keepdim=True)[1]
        direction_idx = pred_direction.max(1, keepdim=True)[1]
        type_idx = pred_type.max(1, keepdim=True)[1]
        pred = torch.cat((color_idx, direction_idx, type_idx), dim=1)
        return pred

    def pre_process(self, image):
        """
        image formatting
        :rtype: PIL.JpegImagePlugin.JpegImageFile
        """
        # image data formatting
        if type(image) == np.ndarray:
            if image.shape[2] == 3:  # turn all 3 channels to RGB format
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 1:  # turn 1 channel to RGB
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # turn numpy.ndarray into PIL.Image
            image = Image.fromarray(image)
        elif type(image) == PIL.JpegImagePlugin.JpegImageFile:
            if image.mode == 'L' or image.mode == 'I':  # turn 8bits or 32bits into 3 channels RGB
                image = image.convert('RGB')

        return image

    def predict(self, img):
        """
        predict vehicle attributes by classifying
        :return: vehicle color, direction and type
        """
        # image pre-processing
        img = self.transforms(img)
        img = img.view(1, 3, 224, 224)

        # put image data into device
        img = img.to(device)

        # calculating inference
        output = self.net.forward(img)

        # get result
        # self.get_predict_ce, return pred to host side(cpu)
        pred = self.get_predict(output)
        color_name = self.color_attrs[pred[0][0]]
        direction_name = self.direction_attrs[pred[0][1]]
        type_name = self.type_attrs[pred[0][2]]

        return color_name, direction_name, type_name
ima=cv2.imread(r"C:\Data\Github\Vehicle-Car-detection-and-multilabel-classification-master\test_imgs\1.jpg")
ima=Image.fromarray(ima)
classifier = Car_Classifier(num_cls=19,
                                         model_path=local_model_path)
car_color, car_direction, car_type = classifier.predict(ima)
print(car_color,car_direction,car_type)