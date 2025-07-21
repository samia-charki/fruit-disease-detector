Fruit Disease Detector 🍎🍌🍇
This is a deep learning project that detects diseases in fruits using a pre-trained ResNet18 model.

Project Structure
training_pyTorch.py: Training script

photo_prediction.py: Prediction script

best_model.pth: Trained model file

.gitignore: File to ignore large files

requirements.txt: List of required libraries

## Requirements

To install the necessary libraries, run:
pip install -r requirements.txt


Usage
To train the model:
python training_pyTorch.py

To run predictions on new images:
python photo_prediction.py

Description
This project uses a ResNet18 model to classify fruit images and detect diseases affecting them. It leverages PyTorch and Torchvision libraries, along with other tools, to load data, train the model, and evaluate performance.

License
This project is open-source and free to use and modify.