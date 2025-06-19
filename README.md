dataset-link - https://www.kaggle.com/datasets/jehanbhathena/weather-dataset?resource=download



# Image Classification with ResNet18 using PyTorch

This project demonstrates an end-to-end image classification pipeline using a pre-trained ResNet18 model in PyTorch. The training is conducted in Google Colab using a dataset extracted from a ZIP file stored on Google Drive.

---

## 📁 Project Structure

```
/content/
├── drive/
│   └── MyDrive/
│       └── train.zip
├── data/
│   └── <unzipped image folders>
```

Each subfolder inside `train.zip` represents a class label and contains images.

---

## ✅ Features

* Mounts Google Drive and unzips training data
* Cleans up MacOS-specific files (e.g., `__MACOSX`, `._*`)
* Loads images with `ImageFolder`
* Applies standard transformations including resizing, normalization
* Loads a pre-trained ResNet18 model
* Fine-tunes all layers on your custom dataset
* Prints loss per batch and epoch
* Shows whether gradients are flowing through the model

---

## 🛠️ Requirements

This code is designed to run in **Google Colab**. Required packages:

* torch
* torchvision
* tqdm
* torchsummary

Install any missing package using pip:

```bash
!pip install torch torchvision tqdm torchsummary
```

---

## 🧠 Model

* **Base Model:** ResNet18 from `torchvision.models`
* **Modifications:** Replaces the final fully connected layer to match the number of classes in the dataset.
* **Loss Function:** Cross Entropy Loss
* **Optimizer:** Adam with learning rate `1e-4`
* **Training Epochs:** 5
* **Batch Size:** 32

---

## 🚀 Training

The model is trained using this loop:

```python
for epoch in range(epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

Progress is displayed using `tqdm`.

---

## 📦 Data Format

The data should be structured like this before zipping:

```
train/
├── class1/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── class2/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── ...
```

Upload `train.zip` to your Google Drive, e.g., `/MyDrive/train.zip`.

---

## 📈 Output

After training, you will see:

* Batch-wise and epoch-wise loss
* Gradient check for each parameter
* Image tensor shape printed once

---

## 📚 License

This project is provided for educational purposes. Modify and use as needed.

