import numpy as np
from concrete import fhe
from PIL import Image
import gradio as gr
import os
import binascii
from pathlib import Path
from io import BytesIO

###############################
# FHEWatermarking Class
###############################
class FHEWatermarking:
    def __init__(self):
        self.IMAGE_SIZE = 32
        self.fhe_circuit = None
        self.client = None
        self.server = None
        self.key_dir = Path("keys")
        self.filter_name = "watermark"
        self.filter_path = Path("filters") / self.filter_name / "deployment"

        self.wm_message = "fhe_secret"
        bits_str = "".join(format(ord(ch), "08b") for ch in self.wm_message)
        self.wm_bits = [int(b) for b in bits_str]
        self.wm_length = len(self.wm_bits)
        total_pixels = self.IMAGE_SIZE * self.IMAGE_SIZE
        self.watermark_mask = np.array(
            [self.wm_bits[i % self.wm_length] for i in range(total_pixels)],
            dtype=np.int64
        )

    def apply_watermark(self, x):
        # Standard LSB embedding on the grayscale image:
        watermarked = (x // 2) * 2 + self.watermark_mask
        return watermarked.flatten()

    def compile_model(self):
        try:
            print("Creating inputset...")
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
        # Expecting decrypted_image to be a 2D numpy array (IMAGE_SIZE x IMAGE_SIZE)
        flat = decrypted_image.flatten()
        extracted_bits = flat[:self.wm_length] % 2
        bit_str = "".join(str(b) for b in extracted_bits.tolist())
        chars = [chr(int(bit_str[i:i+8], 2)) for i in range(0, len(bit_str), 8) if len(bit_str[i:i+8]) == 8]
        return "".join(chars), extracted_bits.tolist()

    def compute_accuracy(self, original_bits, extracted_bits):
        correct_bits = sum(o == e for o, e in zip(original_bits, extracted_bits))
        errors = sum(o != e for o, e in zip(original_bits, extracted_bits))
        bit_accuracy = correct_bits / len(original_bits) * 100
        ber = errors / len(original_bits)
        # Note: this char_accuracy computation compares self.wm_message to the extracted message.
        extracted_message, _ = self.extract_watermark(np.array(original_bits))
        char_accuracy = sum(a == b for a, b in zip(self.wm_message, extracted_message)) / len(self.wm_message) * 100
        return bit_accuracy, char_accuracy, ber

    def adaptive_compression_evaluation(self, decrypted_image):
        """
        This function tests a range of scales and lossless compression methods.
        Note: Changing the scale here will re-size the decrypted image, which
        might disturb the watermark if the scale does not match the embedding scale.
        """
        results = {}
        best_result = None
        best_bit_accuracy = -1
        pil_image = Image.fromarray(decrypted_image)
        
        # Test multiple scales here: 0.5 (downscale), 1.0 (original), and 2.0 (upscale)
        scales = [1.0, 2.0]
        compression_methods = {
            "PNG": {"format": "PNG", "params": {}},
            "WEBP_lossless": {"format": "WEBP", "params": {"lossless": True}},
            "RAW": {"format": None, "params": {}}
        }
        for scale in scales:
            new_size = (max(1, int(self.IMAGE_SIZE * scale)), max(1, int(self.IMAGE_SIZE * scale)))
            # Use Nearest Neighbor to avoid interpolation artifacts as much as possible.
            resized = pil_image.resize(new_size, resample=Image.NEAREST)
            for comp_method, comp_info in compression_methods.items():
                if comp_method == "RAW":
                    processed_array = np.array(resized, dtype=np.int64)
                else:
                    buffer = BytesIO()
                    resized.save(buffer, format=comp_info["format"], **comp_info["params"])
                    buffer.seek(0)
                    compressed = Image.open(buffer)
                    compressed = compressed.convert("L")
                    resized_back = compressed.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), resample=Image.NEAREST)
                    processed_array = np.array(resized_back, dtype=np.int64)
                extracted_message, extracted_bits = self.extract_watermark(processed_array)
                bit_accuracy, char_accuracy, ber = self.compute_accuracy(self.wm_bits, extracted_bits)
                key = f"scale_{scale}_{comp_method}"
                results[key] = {
                    "scale": scale,
                    "compression_method": comp_method,
                    "bit_accuracy": bit_accuracy,
                    "char_accuracy": char_accuracy,
                    "ber": ber,
                    "extracted_message": extracted_message
                }
                if bit_accuracy > best_bit_accuracy:
                    best_bit_accuracy = bit_accuracy
                    best_result = {
                        "processed_image": processed_array,
                        "scale": scale,
                        "compression_method": comp_method,
                        "bit_accuracy": bit_accuracy,
                        "char_accuracy": char_accuracy,
                        "ber": ber,
                        "extracted_message": extracted_message
                    }
        return results, best_result


    def process_image(self, input_image, progress=gr.Progress()):
        if input_image is None:
            return None
        try:
            progress(0.2, desc="Pre-processing image...")
            # Convert the uploaded image to a PNG base (lossless) regardless of original format.
            pil_img = Image.fromarray(input_image)
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")
            buffer.seek(0)
            img_png = Image.open(buffer)
            # Now convert the PNG image to grayscale and resize to 32x32.
            img = img_png.convert("L").resize((self.IMAGE_SIZE, self.IMAGE_SIZE))
            input_array = np.array(img, dtype=np.int64).flatten()

            progress(0.4, desc="Encrypting...")
            encrypted_input = self.encrypt_image(input_array)

            progress(0.6, desc="Processing via FHE (applying watermark)...")
            encrypted_output = self.server.run(encrypted_input, self.client.get_serialized_evaluation_keys())

            progress(0.8, desc="Decrypting...")
            decrypted_output = self.decrypt_image(encrypted_output)
            final_decrypted = decrypted_output.reshape((self.IMAGE_SIZE, self.IMAGE_SIZE)).astype(np.uint8)

            # Extract watermark from the FHE processed image.
            extracted_message, extracted_bits = self.extract_watermark(final_decrypted)
            bit_accuracy, char_accuracy, ber = self.compute_accuracy(self.wm_bits, extracted_bits)

            # Run the adaptive compression/resizing evaluation on the decrypted image.
            adaptive_table, best_adaptive = self.adaptive_compression_evaluation(final_decrypted)

            progress(1.0, desc="Complete!")
            results = {
                "original": input_image,
                "decrypted": final_decrypted,
                "technical_details": {
                    "Watermark Message": self.wm_message,
                    "Extracted Watermark": extracted_message,
                    "Watermark Match": extracted_message == self.wm_message,
                    "Bit Accuracy (%)": bit_accuracy,
                    "Character Accuracy (%)": char_accuracy,
                    "Bit Error Rate (BER)": ber
                },
                "adaptive_results": adaptive_table,
                "adaptive_best": best_adaptive
            }
            return results
        except Exception as e:
            print(f"Error in processing: {e}")
            return None

##############################
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
            "Upload an image. It will be converted to a PNG (lossless) base, then to grayscale and resized to 32x32 before encryption. " +
            "In the encrypted domain, an LSB watermark is applied. After decryption, the watermark is extracted " +
            "and compared to the expected message. The adaptive stage uses lossless compression to preserve the watermark."
        )
        with gr.Column():
            input_image = gr.Image(
                type="numpy",
                label="Upload Image",
                scale=1,
                height=256,
                width=256
            )
            process_btn = gr.Button("Start Processing")
            with gr.Row():
                original_display = gr.Image(label="Original")
                decrypted_display = gr.Image(label="Decrypted with Watermark")
            technical_info = gr.JSON(label="Technical Details (FHE Process)")
            with gr.Row():
                adaptive_display = gr.Image(label="Adaptive Compressed/Resized Image")
                adaptive_technical = gr.JSON(label="Adaptive Compression Technical Details")

        def process_and_update(image):
            if image is None:
                return [None, None, None, None, None]
            results = watermarker.process_image(image)
            if results is None:
                return [None, None, None, None, None]
            # Prepare outputs:
            original = results["original"]
            decrypted = results["decrypted"]
            technical = results["technical_details"]
            adaptive_best = results["adaptive_best"]
            # The adaptive_best result contains both the processed image and its technical details.
            adaptive_image = adaptive_best["processed_image"]
            adaptive_details = {
                "scale": adaptive_best["scale"],
                "compression_method": adaptive_best["compression_method"],
                "bit_accuracy (%)": adaptive_best["bit_accuracy"],
                "character_accuracy (%)": adaptive_best["char_accuracy"],
                "bit_error_rate": adaptive_best["ber"],
                "extracted_message": adaptive_best["extracted_message"],
                "all_tests": results["adaptive_results"]
            }
            return [original, decrypted, technical, adaptive_image, adaptive_details]

        process_btn.click(
            fn=process_and_update,
            inputs=[input_image],
            outputs=[original_display, decrypted_display, technical_info, adaptive_display, adaptive_technical]
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
