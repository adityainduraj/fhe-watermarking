## An Introduction
FHE or Fully Homomorphic Encryption is a method to perform mathematical operations on encrypted data. This guarantees data security and privacy as the data isn't disclosed at any point of time as operations are performed. The result can be seen on decryption. 

## About This Particular Project
It's a basic implementation of FHE using Zama's Concrete and Concrete-ML libraries for FHE for a watermarking system that applies watermarks on encrypted data. I'm developing this as a project to further my understanding of FHE and it's real-life use cases. [Read the technical report.](https://github.com/adityainduraj/fhe-watermarking/blob/main/report.md)

The goal is to develop a system that can perform invisible watermarking operations on encrypted images. This approach is particularly relevant in light of recent developments in Generative AI and regulatory efforts like the EU AI Act, which push for reliable digital watermarking of AI-generated content.

FHE could enable a trustless service that allows standardization across all generated images, addressing the growing need for attribution and traceability in GenAI outputs.

Applications include:
- **Copyright Protection**: Proving ownership.
- **Authentication**: Verify the authenticity of images based on embedded watermarks.
- **Tamper Detection**: Identify and localize manipulations.
- **Digital Media Tracking**: Monitor distribution and usage of images across platforms.

## How To Build & Run This Project
Pretty straightforward.

Just create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

Then install dependencies:
```bash
pip install -r requirements.txt
```
and proceed to run 
```bash
python watermarking.py
```

If you're on NixOS, a `shell.nix` file included. 
Just enter the shell with 
```bash
nix-shell
```
and then proceed with creating the virtual environment and the above steps. 

## Credits
This project is made by using zama.ai 's Concrete and Concrete-ML libraries. Highly recommend checking them out for anything FHE. 
