
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
processor = AutoProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

def batch_loader(file_path, batch_size=32):
    batch_images = []
    batch_questions = []
    batch_labels = []
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(',')
            if len(parts) == 4:
                question = parts[1]
                answer = parts[2]
                image_path = parts[3]

                try:
                    img = Image.open(image_path).resize((384, 384))
                    batch_images.append(img)
                    batch_questions.append(question)
                    ans1 = w2n.word_to_num(answer)
                    batch_labels.append(str(ans1))  # Convert answer to numeric form, assuming this is required

                    if len(batch_images) == batch_size:
                        yield batch_images, batch_questions, batch_labels
                        batch_images, batch_questions, batch_labels = [], [], []
                except IOError:
                    print("Error: File does not appear to exist.")
        if batch_images:  # Yield any remaining data as the last batch
            yield batch_images, batch_questions, batch_labels
            
            
def process_and_predict(images, questions):

    encoding = processor(images, questions, return_tensors="pt", padding=True)
    outputs = model(**encoding)
    return outputs
    
accuracy = []
for images, questions, labels in batch_loader('data/questions_object_preprocessed.txt'):
    outputs = process_and_predict(images, questions)
    print(outputs.logits)
    probabilities = torch.softmax(outputs.logits, dim=1)
    max_prob_indices = torch.argmax(probabilities, dim=1)
    word_list = []
    for i,k in  enumerate(max_prob_indices):
        print(questions[i])
        print("-------------ground truth --------------")
        print(labels[i])
        print("-------------our answer --------------")
        print(model.config.id2label[int(k)])
        word_list.append(model.config.id2label[int(k)])
        matches = sum(1 for x, y in zip(word_list, labels) if x == y)
        probability = matches / len(labels)
        print("-------------batch accuracy --------------")
        accuracy.append(probability)
        print(probability)

average = sum(accuracy) / len(accuracy)
print("-------------average accuracy --------------")
print(average)

    
"""imgs = []
ans = []
ques = []
label = []
#data/questions_object_preprocessed.txt
with open('data/questions_object_preprocessed.txt', 'r', encoding='utf-8') as file:
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
                print(f"Error opening image {image_path}")"""
                

#encoding = processor(image, text, return_tensors="pt", padding=True)
#encoding = processor(imgs, ques, return_tensors="pt", padding=True)
#outputs = model(**encoding)

#print(outputs.logits.size())
"""
word_list =[]
#probabilities = torch.softmax(outputs.logits, dim=1)
#max_prob_indices = torch.argmax(probabilities, dim=1)
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
    print(probability)"""

