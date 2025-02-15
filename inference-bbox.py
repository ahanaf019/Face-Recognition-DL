import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import *
from models import FaceBBoxModel, PersonIdentificationModel


device = 'cuda' if torch.cuda.is_available() else 'cpu'


BATCH_SIZE = 32
IMAGE_SIZE = 224
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10



train_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomAutocontrast(),
    transforms.RandomGrayscale(),
    transforms.RandomInvert(p=0.1),
])

test_data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



plt.figure(figsize=(5*4,2*4))



bbox_model = FaceBBoxModel(hid_dim=512)
bbox_model, _ = load_state('./checkpoints/FaceBBoxModel.pt', bbox_model, None)

feature_extractor = PersonIdentificationModel(hid_dim=512, out_dim=128)
feature_extractor, _ = load_state('./checkpoints/PersonIdentificationModel.pt', feature_extractor, None)


image1 = read_image('./test_images/im1.jpg', size=None)
image2 = read_image('./test_images/im2.jpg', size=None)
image3 = read_image('./test_images/im3.jpg', size=None)
image4 = read_image('./test_images/im4.jpg', size=None)
image5 = read_image('./test_images/im5.jpg', size=None)
image6 = read_image('./test_images/im6.jpg', size=None)
image7 = read_image('./test_images/im7.jpg', size=None)
print(image4.shape)

import torch.nn.functional as F
def get_face_img(img):
    resized_img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

    bbox_model.eval()
    with torch.inference_mode():
        b1 = bbox_model(test_data_transforms(resized_img).unsqueeze(0))

    b1 = b1 * IMAGE_SIZE
    b1 = b1.type(torch.int64)
    # print(b1)

    _, bbox = resize_image_bbox(resized_img, b1[0], (img.shape[1], img.shape[0]))
    # print(bbox)
    x, y, w, h = bbox
    face_image = img[y:y + h, x:x + w]
    return face_image

image1 = cv2.resize(get_face_img(image1), (IMAGE_SIZE, IMAGE_SIZE))
image2 = cv2.resize(get_face_img(image2), (IMAGE_SIZE, IMAGE_SIZE))
image3 = cv2.resize(get_face_img(image3), (IMAGE_SIZE, IMAGE_SIZE))
image4 = cv2.resize(get_face_img(image4), (IMAGE_SIZE, IMAGE_SIZE))
image5 = cv2.resize(get_face_img(image5), (IMAGE_SIZE, IMAGE_SIZE))
image6 = cv2.resize(get_face_img(image6), (IMAGE_SIZE, IMAGE_SIZE))
image7 = cv2.resize(get_face_img(image7), (IMAGE_SIZE, IMAGE_SIZE))


feature_extractor.eval()
with torch.inference_mode():
    d1 = feature_extractor.inference(test_data_transforms(image1).unsqueeze(0))
    d2 = feature_extractor.inference(test_data_transforms(image2).unsqueeze(0))
    d3 = feature_extractor.inference(test_data_transforms(image3).unsqueeze(0))
    d4 = feature_extractor.inference(test_data_transforms(image4).unsqueeze(0))
    d5 = feature_extractor.inference(test_data_transforms(image5).unsqueeze(0))
    d6 = feature_extractor.inference(test_data_transforms(image6).unsqueeze(0))
    d7 = feature_extractor.inference(test_data_transforms(image7).unsqueeze(0))

from torch.nn.functional import cosine_similarity
# def cosine_sim(embedding1, embedding2):
#     # embedding1 = torch.tensor(embedding1).unsqueeze(0)
#     # embedding2 = torch.tensor(embedding2).unsqueeze(0)
#     # print(embedding1.shape, embedding2.shape)
#     return cosine_similarity(embedding1, embedding2).item()


def l2_normalize(embedding):
    return embedding / np.linalg.norm(embedding)

def euclidian_distance(embedding1, embedding2):
    embedding1 = l2_normalize(embedding1)
    embedding2 = l2_normalize(embedding2)
    return np.linalg.norm(embedding1 - embedding2)

print('1 vs 2', euclidian_distance(d1, d2))
print('1 vs 3', euclidian_distance(d1, d3))
print('1 vs 4', euclidian_distance(d1, d4))
print('1 vs 5', euclidian_distance(d1, d5))
print('1 vs 6', euclidian_distance(d1, d6))
print('1 vs 7', euclidian_distance(d1, d7))

print()
print('3 vs 1', euclidian_distance(d3, d1))
print('3 vs 2', euclidian_distance(d3, d2))
print('3 vs 4', euclidian_distance(d3, d4))
print('3 vs 5', euclidian_distance(d3, d5))
print('3 vs 6', euclidian_distance(d3, d6))
print('3 vs 7', euclidian_distance(d3, d7))