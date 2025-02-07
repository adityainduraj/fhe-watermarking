import numpy as np
from concrete import fhe
from PIL import Image
import gradio as gr
import os
import binascii
from pathlib import Path

###############################
# FHEWatermarking Class
###############################
class FHEWatermarking:
    def __init__(self):
        # Use 32x32 images to lessen computational intensity.
        self.IMAGE_SIZE = 32
        self.fhe_circuit = None
        self.client = None
        self.server = None
        self.key_dir = Path("keys")
        self.filter_name = "watermark"
        self.filter_path = Path("filters") / self.filter_name / "deployment"

        # Setup watermark: choose a message and convert to bits (8-bit per char).
        self.wm_message = "fhe_secret"
        bits_str = "".join(format(ord(ch), "08b") for ch in self.wm_message)
        self.wm_bits = [int(b) for b in bits_str]
        self.wm_length = len(self.wm_bits)
        # Build a repeating watermark mask (one watermark bit per pixel) for a 32x32 image.
        total_pixels = self.IMAGE_SIZE * self.IMAGE_SIZE
        self.watermark_mask = np.array(
            [self.wm_bits[i % self.wm_length] for i in range(total_pixels)],
            dtype=np.int64
        )

    def apply_watermark(self, x):
        """
        FHE-circuit function.
        x: flattened image array (32*32 = 1024 elements).
        For each pixel the least significant bit (LSB) is cleared and replaced with
        the corresponding bit from the watermark mask.
        """
        # (x // 2) * 2 clears the LSB. Adding the watermark_mask inserts the watermark bit.
        watermarked = (x // 2) * 2 + self.watermark_mask
        return watermarked.flatten()

    def compile_model(self):
        try:
            print("Creating inputset...")
            # Generate two sample flattened images of size 32x32.
            inputset = [
                np.random.randint(0, 256, size=(self.IMAGE_SIZE * self.IMAGE_SIZE,), dtype=np.int64)
                for _ in range(2)
            ]
            print("Initializing compiler...")
            compiler = fhe.Compiler(self.apply_watermark, {"x": "encrypted"})
            print("Compiling FHE circuit...")
            self.fhe_circuit = compiler.compile(inputset, show_mlir=True)
            print("Compilation complete!")
            self.save_circuit()
            self.client = FHEClient(self.filter_path, self.filter_name, self.key_dir)
            self.server = FHEServer(self.filter_path)
        except Exception as e:
            print(f"Error during compilation: {e}")
            raise

    def save_circuit(self):
        self.filter_path.mkdir(parents=True, exist_ok=True)
        self.fhe_circuit.server.save(self.filter_path / "server.zip", via_mlir=True)
        self.fhe_circuit.client.save(self.filter_path / "client.zip")

    def encrypt_image(self, input_array):
        return self.client.encrypt_serialize(input_array)

    def decrypt_image(self, encrypted_output):
        return self.client.deserialize_decrypt_post_process(encrypted_output)

    def extract_watermark(self, decrypted_image):
        """
        Extract the watermark from the decrypted image.
        Reads the LSB of the first wm_length pixels (flattened order), groups bits into 8-bit bytes,
        and then converts them into characters.
        """
        flat = decrypted_image.flatten()
        extracted_bits = flat[:self.wm_length] % 2  # Get LSB for the first wm_length pixels.
        bit_str = "".join(str(b) for b in extracted_bits.tolist())
        chars = []
        for i in range(0, len(bit_str), 8):
            byte = bit_str[i:i+8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        return "".join(chars)

    def process_image(self, input_image, progress=gr.Progress()):
        if input_image is None:
            return None

        try:
            progress(0.2, desc="Pre-processing image...")
            # Convert input image to grayscale and resize to 32x32.
            img = Image.fromarray(input_image).convert("L").resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            input_array = np.array(img, dtype=np.int64).flatten()

            progress(0.4, desc="Encrypting...")
            encrypted_input = self.encrypt_image(input_array)

            progress(0.6, desc="Processing via FHE (applying watermark)...")
            encrypted_output = self.server.run(encrypted_input, self.client.get_serialized_evaluation_keys())

            progress(0.8, desc="Decrypting...")
            decrypted_output = self.decrypt_image(encrypted_output)
            # Reshape decrypted output to 32x32.
            final_decrypted = decrypted_output.reshape((self.IMAGE_SIZE, self.IMAGE_SIZE)).astype(np.uint8)

            # Extract watermark from the decrypted image.
            extracted_message = self.extract_watermark(final_decrypted)
            wm_match = (extracted_message == self.wm_message)

            progress(1.0, desc="Complete!")
            results = {
                "original": input_image,
                "decrypted": final_decrypted,
                "technical_details": {
                    "Watermark Message": self.wm_message,
                    "Extracted Watermark": extracted_message,
                    "Watermark Match": wm_match
                }
            }
            return results
        except Exception as e:
            print(f"Error in processing: {e}")
            return None

###############################
# FHEClient Class
###############################
class FHEClient:
    def __init__(self, path_dir, filter_name, key_dir=None):
        self.path_dir = path_dir
        self.key_dir = key_dir
        assert path_dir.exists(), f"{path_dir} does not exist. Please specify a valid path."
        self.client = fhe.Client.load(path_dir / "client.zip", self.key_dir)

    def generate_private_and_evaluation_keys(self, force=False):
        self.client.keygen(force)

    def get_serialized_evaluation_keys(self):
        return self.client.evaluation_keys.serialize()

    def encrypt_serialize(self, input_image):
        encrypted_image = self.client.encrypt(input_image)
        return encrypted_image.serialize()

    def deserialize_decrypt_post_process(self, serialized_encrypted_output_image):
        encrypted_output_image = fhe.Value.deserialize(serialized_encrypted_output_image)
        return self.client.decrypt(encrypted_output_image)

###############################
# FHEServer Class
###############################
class FHEServer:
    def __init__(self, path_dir):
        assert path_dir.exists(), f"{path_dir} does not exist. Please specify a valid path."
        self.server = fhe.Server.load(path_dir / "server.zip")

    def run(self, serialized_encrypted_image, serialized_evaluation_keys):
        encrypted_image = fhe.Value.deserialize(serialized_encrypted_image)
        evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
        encrypted_output = self.server.run(encrypted_image, evaluation_keys=evaluation_keys)
        return encrypted_output.serialize()

###############################
# Gradio Interface
###############################
def create_interface():
    watermarker = FHEWatermarking()
    print("Initializing FHE model...")
    watermarker.compile_model()

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# FHE Watermarking with LSB Invisible Watermark\n\n" +
            "Upload an image. It will be converted to grayscale and resized to 32x32, then encrypted. " +
            "In the encrypted domain, an LSB watermark is applied. After decryption the watermark is extracted " +
            "and compared to the expected message."
        )
        with gr.Column():
            input_image = gr.Image(
                type="numpy",
                label="Upload Image",
                scale=1,
                height=256,
                width=256
            )
            process_btn = gr.Button("â–¶ï¸� Start Processing")
            with gr.Row():
                original_display = gr.Image(label="Original")
                decrypted_display = gr.Image(label="Decrypted with Watermark")
            technical_info = gr.JSON(label="Technical Details")

        def process_and_update(image):
            if image is None:
                return [None, None, None]
            results = watermarker.process_image(image)
            if results is None:
                return [None, None, None]
            return [results["original"], results["decrypted"], results["technical_details"]]

        process_btn.click(
            fn=process_and_update,
            inputs=[input_image],
            outputs=[original_display, decrypted_display, technical_info]
        )
    return demo

###############################
# Main
###############################
if __name__ == "__main__":
    try:
        demo = create_interface()
        demo.launch(server_port=7860)
    except Exception as e:
        print(f"Fatal error: {e}")
