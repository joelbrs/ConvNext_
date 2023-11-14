# pip install datasets
# pip install --upgrade pillow
# pip install torch
# pip install torchvision
# pip install tqdm
from datasets import load_dataset
from transformers import AutoImageProcessor, ConvNextForImageClassification
from torchvision.transforms import (Normalize, Compose, RandomResizedCrop, RandomHorizontalFlip, ToTensor)
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
import torch

# carregando o dataset Stanford Dogs Dataset (http://vision.stanford.edu/aditya86/ImageNetDogs/)
url_dataset = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
dataset = load_dataset("imagefolder", data_files=url_dataset, drop_labels=False, split="train")

labels = dataset.features['label'].names
id2Label = {k: v for k, v in enumerate(labels)}
label2Id = {v: k for k, v in enumerate(labels)}

# pré-processamento de imagens para o treinamento do modelo
image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-tiny-224")

# fonte: https://pytorch.org/vision/stable/transforms.html
transforms = Compose([
    RandomResizedCrop(image_processor.size['shortest_edge']),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
])


def train_transforms(examples):
    examples["pixel_values"] = [transforms(image.convert("RGB")) for image in examples["image"]]
    return examples


processed_dataset = dataset.with_transform(train_transforms)

# definindo o modelo, fonte: https://huggingface.co/docs/transformers/v4.35.0/en/model_doc/convnext#transformers.ConvNextForImageClassification
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ConvNextForImageClassification.from_pretrained("facebook/convnext-tiny-224",
                                                       num_labels=len(labels),
                                                       id2label=id2Label,
                                                       label2id=label2Id,
                                                       ignore_mismatched_sizes=True).to(device)

print(processed_dataset[0])


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    label = torch.tensor([example["label"] for example in examples])

    return {"pixel_values": pixel_values, "labels": label}


dataloader = DataLoader(processed_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(5):
    print("Epoch:", epoch)
    correct = 0
    total = 0
    for idx, batch in enumerate(tqdm(dataloader)):
        # move batch to GPU
        batch = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        # forward pass
        outputs = model(pixel_values=batch["pixel_values"],
                        labels=batch["labels"])

        loss, logits = outputs.loss, outputs.logits
        loss.backward()
        optimizer.step()

        # metrics
        total += batch["labels"].shape[0]
        predicted = logits.argmax(-1)
        correct += (predicted == batch["labels"]).sum().item()

        accuracy = correct / total

        if idx % 100 == 0:
            print(f"Loss after {idx} steps:", loss.item())
            print(f"Accuracy after {idx} steps:", accuracy)


repo_name = "joelbrs/convnext-tiny-stanford-dogs"
model.push_to_hub(repo_name)
image_processor.push_to_hub(repo_name)

