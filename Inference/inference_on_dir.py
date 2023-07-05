import os
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from Inference.engineering_example import preprocess, ONNX_MODEL_PATH, TEST_DIR

# Constants
DATA_FILE_NAME = "reid_output.npy"
BATCH_SIZE = 256


def run_onnx_on_dir(onnx_model_path: str, image_dir: str):
    ort_session = ort.InferenceSession(onnx_model_path,
                                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = ort_session.get_inputs()[0].name

    # Get a list of image file names in the directory
    img_names = [img_name for img_name in os.listdir(image_dir) if img_name.endswith(".jpg")]
    img_names.sort()
    img_paths = [os.path.join(image_dir, img_name) for img_name in img_names]

    num_images = len(img_names)
    num_batches = (num_images + BATCH_SIZE - 1) // BATCH_SIZE

    batched_outputs = []
    for batch_num in tqdm(range(num_batches)):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, num_images)
        batch_files = img_paths[start_idx:end_idx]

        batch_images = []
        for image_path in batch_files:
            image = preprocess(image_path)
            batch_images.append(image[0])
        onnx_input = np.stack(batch_images)
        # Run the model inference
        feat_onnx = ort_session.run(None, {input_name: onnx_input})[0]
        batched_outputs.append(feat_onnx)
        print(f"Processed batch {batch_num + 1}/{num_batches}")
    all_outputs = np.concatenate(batched_outputs)
    output_file_path = os.path.join(os.path.basename(image_dir), DATA_FILE_NAME)
    np.save(output_file_path, {"file_names": img_names, "feat_vec": all_outputs})
    print(f"Saved the outputs to {output_file_path}.")


if __name__ == '__main__':
    run_onnx_on_dir(ONNX_MODEL_PATH, TEST_DIR)
