"""
ViLT Model train V1.0
"""
 
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
 
image1 = Image.open('./1.png').resize((384, 384))
text1 = "what is this people doing?"
label1 = "sitting"
image2 = Image.open('./1.png').resize((384, 384))
text2 = "is this people beautiful? "
label2 = "yes"
image3 = Image.open('./1.png').resize((384, 384))
text3 = "how old is this people?"
label3 = "30"
 

text = [text1, text2, text3]
image = [image1, image2, image3]
label = [label1, label2, label3]
 
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa").cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
 
 
class MyData(Dataset):
    def __init__(self, text, image, label):
        super(MyData, self).__init__()
        self.text = text
        self.image = image
        self.label = label
 

    def __getitem__(self, item):
        text = self.text[item]
        image = self.image[item]
        transform = transforms.ToTensor()
        image = transform(image)
        label = self.label[item]
 
        return text, image, label
 
    def __len__(self):
        return len(self.label)
 
 
#
dataset = MyData(text, image, label)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)
#
 
 
for epoch in range(10):
    model.train()
    train_loss = 0.0
    val_loss = 0.0
    train_acc = 0.0
    val_acc = 0.0

    for i, (text, image, label) in enumerate(dataloader):
        text_list = list(text)
        label = list(label)

        image_list = []
        label_list = []
 
        for idx in label:
            each_label = model.config.label2id[idx]
            label_list.append(each_label)
 
        for n in range(len(image[:, 0, 0])):
            each_image = image[n, :, :]
            image_list.append(each_image)
 
        labels = torch.tensor(label_list)
  
        encoding = processor(image_list, text_list, return_tensors="pt", padding=True)

        labels = labels.cuda()
        for feature, data in encoding.data.items():
            encoding.data[feature] = data.cuda()
 
        outputs = model(**encoding)
        logits = outputs.logits
 
        optimizer.zero_grad()
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(label)
        ret, predictions = torch.max(logits.data, 1)
        correct_counts = predictions.eq(labels.data.view_as(predictions))
        acc = torch.mean(correct_counts.type(torch.FloatTensor))
        train_acc += acc.item() * len(label)

    avg_train_loss = train_loss / len(dataset)
    avg_train_acc = train_acc / len(dataset)
    print(avg_train_loss, avg_train_acc)