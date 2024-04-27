
from transformers import AutoProcessor, ViltForQuestionAnswering
import torch

from PIL import Image


image1 = Image.open('./1.png').resize((384, 384))
text1 = "what is this people doing?"
image2 = Image.open('./1.png').resize((384, 384))
text2 = "is this people beautiful?"
image3 = Image.open('./1.png').resize((384, 384))
text3 = "how old is this people?"
 
text = [text1, text2, text3]
image = [image1, image2, image3]


processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
encoding = processor(image, text, return_tensors="pt", padding=True)
outputs = model(**encoding)

print(outputs.logits.size())

word_list =[]
probabilities = torch.softmax(outputs.logits, dim=1)
max_prob_indices = torch.argmax(probabilities, dim=1)
print(max_prob_indices)
for i in max_prob_indices:
    print(model.config.id2label[int(i)])

