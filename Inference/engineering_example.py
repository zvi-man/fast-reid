import os
import tqdm
import cv2
import faiss
import numpy as np
import torch
import torch.nn.functional as F
import onnxruntime
from typing import List, Tuple

# Constants
TEST_DIR = r"/Inference/test_car"
ONNX_MODEL_PATH = "outputs/onnx_model/VeriWildPreTrained.onnx"
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def preprocess(image_path: str) -> np.ndarray:
    original_image = cv2.imread(image_path)
    # the model expects RGB inputs
    original_image = original_image[:, :, ::-1]

    # Apply pre-processing to image.
    img = cv2.resize(original_image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    img = img.astype("float32").transpose(2, 0, 1)[np.newaxis]  # (1, 3, h, w)
    return img


def normalize(nparray, order=2, axis=-1):
    """Normalize a N-D numpy array along the specified axis."""
    norm = np.linalg.norm(nparray, ord=order, axis=axis, keepdims=True)
    return nparray / (norm + np.finfo(np.float32).eps)


@torch.no_grad()
def compute_cosine_distance(features, others):
    """Computes cosine distance.
    Args:
        features (torch.Tensor): 2-D feature matrix.
        others (torch.Tensor): 2-D feature matrix.
    Returns:
        torch.Tensor: distance matrix.
    """
    features = F.normalize(features, p=2, dim=1)
    others = F.normalize(others, p=2, dim=1)
    dist_m = 1 - torch.mm(features, others.t())
    return dist_m.cpu().numpy()


def engineering_example():
    # Preprocess all images in dir
    img_paths = [os.path.join(TEST_DIR, x) for x in os.listdir(TEST_DIR) if x.endswith(".jpg")]
    img_paths.sort()
    image_list = []
    for path in img_paths:
        print(path)
        image = preprocess(path)
        image_list.append(image[0])
    onnx_input = np.stack(image_list)

    # Run ONNX model
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH,
                                               providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name
    feat_onnx = ort_session.run(None, {input_name: onnx_input})[0]
    feat_onnx = torch.tensor(feat_onnx)

    # Post Process - Calculate distance between images
    dist_mat = compute_cosine_distance(feat_onnx, feat_onnx)
    ranking_mat = np.argsort(dist_mat, axis=1)
    print(ranking_mat)


if __name__ == '__main__':
    engineering_example()
