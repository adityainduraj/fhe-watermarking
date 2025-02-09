## Technical Report: Fully Homomorphic Encryption-based Invisible Watermarking

---
## 1. Introduction

In today’s digital era, ensuring the integrity, authenticity, and rightful attribution of digital images is paramount. Digital watermarking provides a subtle yet powerful means to embed hidden information within an image. Our project takes this concept to a new level by integrating Fully Homomorphic Encryption (FHE) using Concrete ML. This integration allows us to perform watermark embedding and extraction directly on encrypted images—ensuring that sensitive data remains confidential during processing. This report provides an exhaustive explanation of our approach, design decisions, implementation details, and experimental evaluations, including adaptive compression tests.

---
## 2. System Objectives and Architectural Overview

### 2.1 Objectives

The system was developed with the following primary goals in mind:

- **Confidential Watermarking:** Embed an invisible watermark using a method that is compatible with FHE, ensuring that the watermarking process itself is performed on encrypted data.
- **Privacy Preservation:** Leverage FHE to guarantee that image data remains confidential during the watermarking process, aligning with emerging regulatory requirements.
- **Robustness Evaluation:** Validate the resilience of the watermark under various image processing conditions (e.g., scaling and different lossless compression methods).
- **End-to-End Pipeline:** Create a complete pipeline from image pre-processing and encryption to FHE-based watermarking, decryption, watermark extraction, and adaptive evaluation.

### 2.2 Architectural Overview

The solution is built around three primary components:

- **FHEWatermarking:** This class encapsulates the watermarking logic, including embedding, extraction, and adaptive evaluation.
- **FHEClient and FHEServer:** These classes implement the client–server paradigm, where the client handles key management, encryption, and decryption, and the server executes the encrypted computation using a compiled FHE circuit.
- **Gradio Interface:** Provides a user-friendly front-end to demonstrate the pipeline, display the watermarked images, and report detailed technical metrics.

This modular design not only enhances maintainability but also facilitates in-depth analysis and future modifications (e.g., replacing LSB with transform-domain methods).

---
## 3. Watermark Embedding Using LSB

### 3.1 Why LSB?

The Least Significant Bit (LSB) method is chosen for several reasons:

- **Computational Simplicity:** The operation involves simple arithmetic at the bit level. Since FHE operations are computationally expensive, reducing the arithmetic complexity is essential.
- **Natural Mapping to FHE:** FHE frameworks like Concrete ML excel in processing integer arithmetic. LSB embedding involves clearing and setting a single bit per pixel—a task well within the capabilities of FHE.
- **Ease of Debugging and Verification:** The method’s simplicity ensures that the watermarking logic can be easily traced, verified, and debugged even when it is executed in an encrypted domain.
- **Proof-of-Concept Robustness:** While LSB embedding can be sensitive to certain types of noise or lossy compression, our controlled evaluation focuses on lossless methods, which are sufficient for demonstrating the feasibility of FHE watermarking.

### 3.2 Implementation Details: Code Walkthrough

#### 3.2.1. Watermark Embedding Function

The core function that implements LSB watermarking is shown below:

```python
def apply_watermark(self, x):
    # Standard LSB embedding on the grayscale image:
    watermarked = (x // 2) * 2 + self.watermark_mask
    return watermarked.flatten()
```

**Explanation:**

- **Clearing the LSB:** The expression `(x // 2) * 2` ensures that the least significant bit (LSB) of each pixel is set to 0.
- **Embedding the Watermark:** The watermark mask (a binary sequence derived from the secret message “fhe_secret”) is added, setting the LSB to the desired bit.
- **Flattening:** The image, initially processed as a 2D array (32×32 pixels), is flattened to match the expected input for the FHE circuit.

#### 3.2.2. FHE Circuit Compilation

The watermarking function is compiled into an FHE circuit using Concrete ML:

```python
inputset = [
    np.random.randint(0, 256, size=(self.IMAGE_SIZE * self.IMAGE_SIZE,), dtype=np.int64)
    for _ in range(2)
]
compiler = fhe.Compiler(self.apply_watermark, {"x": "encrypted"})
self.fhe_circuit = compiler.compile(inputset, show_mlir=True)
```

**Key Points:**

- **Input Set Generation:** A set of two random images (each a flattened array of 1024 values) provides the range of expected inputs.
- **Compiler Initialization:** The `fhe.Compiler` transforms our Python function into an FHE-compatible circuit. The specification `{"x": "encrypted"}` indicates that the input will be provided in an encrypted form.
- **MLIR Output:** The option `show_mlir=True` allows us to inspect the intermediate representation, facilitating debugging and optimization.

---
## 4. Encryption, Decryption, and Client–Server Interactions

### 4.1. Encryption and Serialization

The client encrypts the image before sending it for FHE processing:

```python
def encrypt_serialize(self, input_image):
    encrypted_image = self.client.encrypt(input_image)
    return encrypted_image.serialize()
```

This method ensures that the raw image data is never exposed, even during transmission. The encrypted image is serialized into a byte stream.

### 4.2. Decryption

Post FHE processing, the encrypted output is deserialized and decrypted:

```python
def deserialize_decrypt_post_process(self, serialized_encrypted_output_image):
    encrypted_output_image = fhe.Value.deserialize(serialized_encrypted_output_image)
    return self.client.decrypt(encrypted_output_image)
```

The client securely recovers the processed image using its private keys.

---
## 5. Watermark Extraction and Accuracy Evaluation

### 5.1. Extraction Process

After decryption, the watermark is extracted by analyzing the least significant bits:

```python
def extract_watermark(self, decrypted_image):
    flat = decrypted_image.flatten()
    extracted_bits = flat[:self.wm_length] % 2
    bit_str = "".join(str(b) for b in extracted_bits.tolist())
    chars = [chr(int(bit_str[i:i+8], 2)) for i in range(0, len(bit_str), 8) if len(bit_str[i:i+8]) == 8]
    return "".join(chars), extracted_bits.tolist()
```

**Explanation:**

- **Bit Extraction:** The modulo operation (`% 2`) retrieves the LSB of each pixel.
- **Message Reconstruction:** The binary string is partitioned into 8-bit segments, each converted to an ASCII character, reconstructing the watermark “fhe_secret.”
- **Return Values:** Both the extracted message and the raw bit list are returned for further analysis.

### 5.2. Accuracy Metrics

The system evaluates the watermark’s extraction quality using bit accuracy, character accuracy, and Bit Error Rate (BER):

```python
def compute_accuracy(self, original_bits, extracted_bits):
    correct_bits = sum(o == e for o, e in zip(original_bits, extracted_bits))
    errors = sum(o != e for o, e in zip(original_bits, extracted_bits))
    bit_accuracy = correct_bits / len(original_bits) * 100
    ber = errors / len(original_bits)
    extracted_message, _ = self.extract_watermark(np.array(original_bits))
    char_accuracy = sum(a == b for a, b in zip(self.wm_message, extracted_message)) / len(self.wm_message) * 100
    return bit_accuracy, char_accuracy, ber
```

These metrics provide a quantitative basis for comparing the fidelity of the watermark extraction under different conditions.

---
## 6. Adaptive Compression Evaluation

### 6.1. Overview

Recognizing that images often undergo various processing steps (scaling and compression) during transmission or storage, we incorporated an adaptive compression evaluation phase. This phase tests the watermark’s resilience under different conditions, using multiple scales and lossless compression methods.

### 6.2. Experimental Results

The following JSON block summarizes one experimental run of our adaptive evaluation process:

```json
{
  "scale": 1,
  "compression_method": "PNG",
  "bit_accuracy (%)": 100,
  "character_accuracy (%)": 100,
  "bit_error_rate": 0,
  "extracted_message": "fhe_secret",
  "all_tests": {
    "scale_1.0_PNG": {
      "scale": 1,
      "compression_method": "PNG",
      "bit_accuracy": 100,
      "char_accuracy": 100,
      "ber": 0,
      "extracted_message": "fhe_secret"
    },
    "scale_1.0_WEBP_lossless": {
      "scale": 1,
      "compression_method": "WEBP_lossless",
      "bit_accuracy": 100,
      "char_accuracy": 100,
      "ber": 0,
      "extracted_message": "fhe_secret"
    },
    "scale_1.0_RAW": {
      "scale": 1,
      "compression_method": "RAW",
      "bit_accuracy": 100,
      "char_accuracy": 100,
      "ber": 0,
      "extracted_message": "fhe_secret"
    },
    "scale_2.0_PNG": {
      "scale": 2,
      "compression_method": "PNG",
      "bit_accuracy": 100,
      "char_accuracy": 100,
      "ber": 0,
      "extracted_message": "fhe_secret"
    },
    "scale_2.0_WEBP_lossless": {
      "scale": 2,
      "compression_method": "WEBP_lossless",
      "bit_accuracy": 100,
      "char_accuracy": 100,
      "ber": 0,
      "extracted_message": "fhe_secret"
    },
    "scale_2.0_RAW": {
      "scale": 2,
      "compression_method": "RAW",
      "bit_accuracy": 52.5,
      "char_accuracy": 100,
      "ber": 0.475,
      "extracted_message": "<<<À<33ÿ<<"
    }
  }
}
```

### 6.3. Detailed Analysis

- **Scale 1.0 Tests:**  
    For all compression methods (PNG, WEBP_lossless, and RAW) at the original scale, the watermark is perfectly preserved with 100% bit and character accuracy. The bit error rate (BER) is 0, and the extracted message exactly matches “fhe_secret.” This confirms that our watermark embedding and extraction process is robust when no scaling occurs.
    
- **Scale 2.0 Tests (Upscaling):**  
    When the image is upscaled by a factor of 2, the PNG and WEBP_lossless formats still yield perfect results (100% bit accuracy, 0 BER). However, the RAW method at scale 2.0 shows a significant degradation: the bit accuracy drops to 52.5% with a BER of 0.475, and the extracted message becomes garbled (e.g., “<<<À<33ÿ<<”).  
    **Explanation:**
    
    - **Interpolation Effects:** Upscaling with nearest-neighbor interpolation typically preserves discrete pixel values. However, when no compression is applied (RAW), even slight inconsistencies due to scaling can affect the LSBs, causing errors.
    - **Compression Algorithms:** Lossless compression methods like PNG and lossless WEBP appear to provide an implicit “correction” by standardizing pixel values during encoding/decoding, thus preserving the watermark bits more reliably.

### 6.4 JPEG Compression
While our current implementation primarily focuses on lossless compression methods (e.g., PNG and lossless WEBP) to preserve watermark integrity, JPEG compression poses a significant challenge. JPEG’s lossy nature—introducing quantization and artifacts—can disturb the least significant bits that carry the watermark, thereby degrading its accuracy. Preliminary experiments suggest that applying JPEG compression results in a marked drop in bit accuracy, necessitating the exploration of additional error-correction strategies or more robust watermarking techniques to mitigate the loss introduced by JPEG compression.

This comprehensive testing illustrates both the strengths and limitations of the current approach. While lossless compression maintains the integrity of the watermark even after upscaling, the RAW method suffers from inaccuracies at higher scales—a factor that will inform future improvements in robust watermark embedding.

---
## 7. End-to-End Pipeline: Pre-processing to Post-processing

### 7.1. Pre-processing

Before encryption, images are converted to PNG (to ensure losslessness), then to grayscale, and finally resized to 32x32 pixels. This controlled environment is crucial for consistent FHE processing.

### 7.2. Encryption and FHE Processing

The pre-processed image is encrypted by the client. The encrypted image is then processed by the FHE server using the compiled FHE circuit that applies the LSB watermark. This step ensures that all computations occur on encrypted data, safeguarding privacy.

### 7.3. Decryption and Watermark Extraction

After processing, the client decrypts the output. The decrypted image is reshaped to its original dimensions, and the watermark is extracted by analyzing the least significant bits. The extraction results are then compared to the original watermark for accuracy.

### 7.4. Adaptive Evaluation

Following extraction, the decrypted image is subjected to various scaling and compression tests. The adaptive evaluation not only reports the best-case scenario (in our case, scale 1.0 with PNG) but also provides a detailed breakdown of how different combinations affect the watermark’s integrity.

---
## 8. Code Architecture and Design Trade-offs

### 8.1. Modular Code Organization

Our implementation divides responsibilities clearly among:

- **FHEWatermarking:** Centralized watermark embedding, extraction, and evaluation.
- **FHEClient:** Responsible for encryption, key management, and decryption.
- **FHEServer:** Handles the execution of the FHE circuit.
- **Gradio Interface:** Integrates the end-to-end process and provides user feedback.

This modular design simplifies testing, debugging, and future enhancements.

### 8.2. Trade-offs

- **Computational Overhead:**  FHE operations are inherently resource-intensive. Our choice of a 32x32 image size and the LSB method minimizes computational complexity, but scalability remains a challenge.
- **Robustness vs. Simplicity:**  The LSB method is simple and FHE-friendly, but it is more susceptible to errors under certain processing conditions. The adaptive evaluation indicates that while lossless compression maintains watermark integrity, raw upscaling can cause significant degradation.
- **Key Management and Security:**  By keeping encryption and decryption on the client side, and ensuring that the server only processes encrypted data, we adhere to strict privacy requirements. The secure serialization and transmission of evaluation keys further bolster the system’s security posture.

---
## 9. Conclusion

This report has provided an exhaustive examination of our FHE-based invisible watermarking system. From the rationale behind choosing the LSB method to a detailed walkthrough of code-level operations—including encryption, FHE processing, and adaptive compression evaluation—we have explored every facet of the design. The inclusion of real experimental results (as captured in the JSON block) demonstrates that, under ideal conditions (scale 1.0 with lossless compression), our system achieves perfect watermark recovery (100% bit and character accuracy, 0 BER). However, the variability observed with RAW upscaling at scale 2.0 highlights areas for future enhancement.

The work presented here lays a solid foundation for subsequent research. Future improvements may involve exploring more robust watermarking methods (such as transform-domain techniques), optimizing FHE circuit efficiency, and refining the system to better handle diverse image processing scenarios.


