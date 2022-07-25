# imports
import numpy as np
import torch
import cv2
from torchvision import transforms

from typing import Type, Tuple, Dict
from torchvision import transforms
from deepface import DeepFace

from network_weight import UNet


def define_model(pretrained_state_dict_path: str = 'models/model_ep_37.pth.tar') -> Type[UNet]:
  """
  Reading and preprocess the model file.

  :pretrained_state_dict_path (incl default path): path to the pretrained model state

  :return: model class with pretrained state dict applied
  """
  model_w = UNet(128, 32, 32)
  pretrained_model_w = torch.load(pretrained_state_dict_path, map_location=torch.device('cpu'))
  model_w.load_state_dict(pretrained_model_w["state_dict"])

  if torch.cuda.is_available():
    model = model_w.cuda()
  else:
    model = model_w
  
  return model

def prepare_image(img_path: str) -> np.ndarray:
  """
  Making an image to the right format to analize with the weight model.

  :img_path: as an input, reading this to numpy.ndarray

  :return: transformed image 
  """
  RES = 128

  X = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB).astype('float32')
  scale = RES / max(X.shape[:2])
    
  X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
    
  if X_scaled.shape[1] > X_scaled.shape[0]:
    p_a = (RES - X_scaled.shape[0])//2
    p_b = (RES - X_scaled.shape[0])-p_a
    X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
  elif X_scaled.shape[1] <= X_scaled.shape[0]:
    p_a = (RES - X_scaled.shape[1])//2
    p_b = (RES - X_scaled.shape[1])-p_a
    X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
    
  o_img = X.copy()
  X /= 255
  X = transforms.ToTensor()(X).unsqueeze(0)
        
  if torch.cuda.is_available():
    X = X.cuda()
    
  return X

def get_gender_and_age(img_path: str) -> Tuple[str, str]:
  """
  Gets gender and age given image path with deepface library.

  :img_path: path to an image

  :return: tuple contains gender and age
  """
  obj = DeepFace.analyze(img_path, actions=['age', 'gender'])

  return (obj['gender'], str(obj['age']))

def predict_result(imgpath : str, model: Type[UNet]) -> Dict:
  """
  Main prediction function.

  :imgpath: path to an image
  :model: weights file (.pth) with UNet architecture described in network_weight.py

  :return: dict with img path, weight, gender and age
  """
  img = prepare_image(imgpath)
  with torch.no_grad():
    _, _, _, w_p = model(img)
  
  weight = w_p
  gender, age = get_gender_and_age(imgpath)

  return {
    "img": imgpath,
    "weight": 100*weight.item(),
    "gender": gender,
    "age": age,
  }