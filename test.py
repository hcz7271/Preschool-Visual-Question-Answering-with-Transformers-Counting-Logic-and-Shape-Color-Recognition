
from transformers import AutoProcessor, ViltForQuestionAnswering
import torch

from PIL import Image
from word2number import w2n

"""
image1 = Image.open('./1.png').resize((384, 384))
text1 = "what is this people doing?"
image2 = Image.open('./1.png').resize((384, 384))
text2 = "is this people beautiful?"
image3 = Image.open('./1.png').resize((384, 384))
text3 = "how old is this people?"
 
text = [text1, text2, text3]
image = [image1, image2, image3]"""

imgs = []
ans = []
ques = []
label = []
#data/questions_object_preprocessed.txt
with open('t.txt', 'r', encoding='utf-8') as file:
    for line in file:
        parts = line.strip().split(',')
        if len(parts) == 4:
            question = parts[1]
            answer = parts[2]
            image_path = parts[3]

            try:
                img = Image.open(image_path).resize((384, 384))
                imgs.append(img)
                ques.append(question)
                ans1 = w2n.word_to_num(answer)
                label.append(str(ans1))
            except IOError:
                print(f"Error opening image {image_path}")
                

processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
#encoding = processor(image, text, return_tensors="pt", padding=True)
encoding = processor(imgs, ques, return_tensors="pt", padding=True)
outputs = model(**encoding)

print(outputs.logits.size())

word_list =[]
probabilities = torch.softmax(outputs.logits, dim=1)
max_prob_indices = torch.argmax(probabilities, dim=1)
print(max_prob_indices)
for i,k in  enumerate(max_prob_indices):
    print(ques[i])
    print("-------------ground truth --------------")
    print(label[i])
    print("-------------our answer --------------")
    print(model.config.id2label[int(k)])
    word_list.append(model.config.id2label[int(k)])

print(label)
print(word_list)
if len(label) != len(word_list):
    print("Lists are of unequal length.")
else:
    matches = sum(1 for x, y in zip(label, word_list) if x == y)
    probability = matches / len(label)
    print("-------------accuracy --------------")
    print(probability)

