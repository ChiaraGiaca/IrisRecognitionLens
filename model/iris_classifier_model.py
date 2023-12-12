import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import models, transforms
from torchvision.datasets.folder import default_loader
from typing import List, Tuple
from keras import backend as K
from model.user import User

# Data augmentation for training and validation
DATA_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=90),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ColorJitter(), #nuovo
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def calc_euclidean(self, x1, x2):
        return torch.norm(x1-x2,2)

    def cosine_similarity(self, x1, x2):
      cos = nn.CosineSimilarity(dim=1, eps=1e-6)
      return cos(x1, x2)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses

class IrisClassifier(nn.Module):
    CHECKPOINT_FILE_NAME = "iris_recognition_trained_model-2.pt"


    def __init__(self,
                 class_names: List[str] = None,
                 num_classes: int = 50,
                 load_from_checkpoint: bool = False,
                 acceptance_threshold: float = 0.50,
                 image_loader=default_loader,
                 checkpoint_file: str = CHECKPOINT_FILE_NAME):
        super().__init__()
        assert class_names is not None or load_from_checkpoint, \
            "Either load a model with predefined classes from a checkpoint, " \
            "or provide them up front"

        if class_names is not None:
            assert len(class_names) == num_classes, \
                "Number of classes must be equal to the length of class " \
                "names provided"
            self.class_names: List[str] = sorted(class_names)
            self.num_classes = len(self.class_names)
        else:
            self.num_classes = num_classes

        self.device = torch.device("cuda:0" if torch.cuda.is_available()
                                   else "cpu")

        #chosen model after ablation study
        model = models.densenet121(pretrained=True)
        
        # Replace the last ResNet layer to match the desired number of classes
        num_features = model.classifier.in_features
        #de comment it if you don't want a siamese network
        model.classifier = nn.Linear(num_features, self.num_classes)

        #added for siamese network 
        #comment it if you don't want a siamese network
        '''self.classifier = nn.Sequential(
            nn.Linear(3000, num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, self.num_classes),
            nn.Dropout(p=0.2)
        )
       
        self.classifier = self.classifier.to(self.device)
        self.sigmoid = nn.Sigmoid().to(self.device)'''
        ###comment until here"""
      
        model = model.to(self.device)
        self.model: nn.Module = model

        #loading model parameters if given
        if load_from_checkpoint:
            checkpoint = torch.load(checkpoint_file, map_location=self.device)
            state_dict = checkpoint["model_state_dict"]

            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                # Local fix when loading model that was saved with DataParallel
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("model.", "")
                    new_state_dict[name] = v
                self.model.load_state_dict(new_state_dict)
                
            self.class_names = checkpoint["classes"]

        self.acceptance_threshold = acceptance_threshold
        self.image_loader = image_loader
        self.transform = DATA_TRANSFORMS["val"]

    #added for tripletloss
    '''def forward_once(self, x):
        output = self.densenet(x)
        output = output.view(output.size()[0], -1)
        return output'''

    def forward(self, x):
        output = self.model(x) 
        return output

    #function used to apply the model to a single image e return probabilities for each labels, in a normal and sorted way
    def classify_single_image(self,
                              normalized_image_path: str):
        """
        Predict a class and determine its probability
        """
        self.model.eval()
        image = self.image_loader(normalized_image_path)
        image = self.transform(image).float().to(self.device)
        image = image.unsqueeze_(0)
       
        with torch.set_grad_enabled(False):
            output = self.model(image)
            output=nn.functional.softmax(output,dim=1).squeeze()
            index = output.cpu().numpy().argmax()
            list_output = output.cpu().numpy().tolist()
            predicted_class = self.class_names[index]
            prob_labels=[]
            for i in range(len(list_output)):
              prob_labels.append((list_output[i],self.class_names[i]))
            sorted_labels = [x for _,x in sorted(prob_labels, reverse= True)]

        return predicted_class, sorted_labels, sorted(list_output,reverse=True), self.class_names, list_output
