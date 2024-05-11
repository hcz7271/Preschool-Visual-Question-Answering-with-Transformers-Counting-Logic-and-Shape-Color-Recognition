from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
from torchvision import transforms
from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
from word2number import w2n
import os

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torch.optim.lr_scheduler import CosineAnnealingLR

import argparse
import ot


def sinkhorn_distance(x, y, epsilon=1.0, num_iters=20):
    """

    Args:
        x (torch.Tensor): Tensor of shape (batch_size, num_features_x, feature_dim)
        y (torch.Tensor): Tensor of shape (batch_size, num_features_y, feature_dim)
        epsilon (float): 
        num_iters (int): Sinkhorn 

    Returns:
        torch.Tensor:
    """
    x_col = x.unsqueeze(2)  # (batch_size, num_features_x, 1, feature_dim)
    y_lin = y.unsqueeze(1)  # (batch_size, 1, num_features_y, feature_dim)
    C = torch.sum((x_col - y_lin) ** 2, dim=-1)  # (batch_size, num_features_x, num_features_y) 32 9 144
    
    C = C / C.max()
    #  e^(-C/epsilon) 32 9 144
    K = torch.exp(-C / epsilon)
    log_K = -C / epsilon

    # a, b means
    a = torch.ones(x.size(0), x.size(1), device=x.device) / x.size(1)
    b = torch.ones(y.size(0), y.size(1), device=y.device) / y.size(1)

    # Sinkhorn 
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for _ in range(num_iters):
        log_u = torch.log(a) - torch.logsumexp(log_K + log_v.unsqueeze(1), dim=2)
        log_v = torch.log(b) - torch.logsumexp(log_K.transpose(-2, -1) + log_u.unsqueeze(1), dim=2)


    log_T = log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(-2)
    T = torch.exp(log_T) 


    distance = torch.sum(T * C, dim=(-2, -1))

    return distance



class ConsistencyClassifier(nn.Module):
    def __init__(self, hidden_size):
        super(ConsistencyClassifier, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dense(x)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.sigmoid(x)
        return x

        

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
                    batch_labels.append(str(ans1))

                    if len(batch_images) == batch_size:
                        yield batch_images, batch_questions, batch_labels
                        batch_images, batch_questions, batch_labels = [], [], []
                except IOError:
                    print(f"Error opening image {image_path}")
        if batch_images:  # Yield any remaining data as the last batch
            yield batch_images, batch_questions, batch_labels


def process_and_predict(images, questions, processor, model):

    encoding = processor(images, questions, return_tensors="pt", padding=True)
    outputs = model(**encoding)
    return outputs




def valid(model, best_acc, processor, args):
    print("--------------------this is validation------------------------")
    accuracy = []
    model.cpu()
    model.eval()
    for images, questions, labels in batch_loader('data/val1_data.txt'):#ata/val_data
        outputs = process_and_predict(images, questions, processor, model)
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
    if average> best_acc:
        best_acc = average 
        model.save_pretrained(args.output_dir)
        print("save in:{save_path}")
        
    
def get_args():
    parser = argparse.ArgumentParser(description="Train a VQA model with command line arguments.")
    parser.add_argument('--train_data', type=str, default='data/train1_data.txt', help='Path to the training data file.')
    parser.add_argument('--val_data', type=str, default='data/val1_data.txt', help='Path to the validation data file.')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for optimizer.')# 0.00005
    parser.add_argument('--output_dir', type=str, default='checkpoint/', help='Directory to save outputs.')
    parser.add_argument('--beta', type=int, default=0.001, help='hyperpar of ConsistencyClassifier')
    parser.add_argument('--theta', type=int, default=0.000005, help='hyperpar of ConsistencyClassifier')

    return parser.parse_args()

def main():
    args = get_args()
    processor = ViltProcessor.from_pretrained("/data/ryh/xianyu/vilt/")
    model = ViltForQuestionAnswering.from_pretrained("/data/ryh/xianyu/vilt/").cuda()
    loss_function = torch.nn.CrossEntropyLoss()
    criteria = nn.BCELoss()
    cls = ConsistencyClassifier(hidden_size=768)
    optimizer = torch.optim.Adam(list(cls.parameters())+list(model.parameters()), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)
    
    
    for epoch in range(args.epochs):
        model.cuda()
        cls.cuda()
        cls.train()
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        avg_train_loss = []
        avg_train_acc = []
        num = 0

        for images, questions, labels in batch_loader('data/train1_data.txt', batch_size=32):
        
        #image_list = [transforms.ToTensor()(image) for image in images]
            label_list = [model.config.label2id[label] for label in labels]
            labels_tensor = torch.tensor(label_list).cuda()
        
    
        
       
            encoding = processor(images, questions, return_tensors="pt", padding=True, truncation=True)
            for feature, data in encoding.data.items():
                encoding.data[feature] = data.cuda()

       
            #outputs = model(**encoding)
            
            outputs = model.vilt(**encoding)
            text_feature = outputs.last_hidden_state[:, 1:10, :]
            image_feature = outputs.last_hidden_state[:, -144:, :]
            
            
            #SinkhornDistance
            normalized_text_features = text_feature / text_feature.sum(dim=1, keepdim=True)
            normalized_image_features = image_feature / image_feature.sum(dim=1, keepdim=True)
            
            # 
            OT_distance = sinkhorn_distance(normalized_text_features, normalized_image_features)
            loss2 = OT_distance.mean()
            
            


            logits = model.classifier(outputs.pooler_output)
            labels = torch.FloatTensor([[1] for i in range(len(labels))])
            loss1 = criteria(cls(outputs.pooler_output), labels.cuda())
            
            
            #logits = outputs.logits
            optimizer.zero_grad()
            loss = loss_function(logits, labels_tensor) + args.beta*loss1 + args.theta*loss2
            loss.backward()
            optimizer.step()
            num += len(labels)
       

        
            train_loss += loss.item() * len(labels)
            ret, predictions = torch.max(logits.data, 1)
            correct_counts = predictions.eq(labels_tensor.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * len(labels)
        
            avg_train_loss.append(train_loss)
            avg_train_acc.append(train_acc)
        

        avg_train_loss1 = sum(avg_train_loss) / num
        avg_train_acc1 = sum(avg_train_acc) / num
        print(f"Epoch {epoch+1}, Average Training Loss: {avg_train_loss1}, Average Training Accuracy: {avg_train_acc1}")
    
        scheduler.step()
        best_acc=0
        valid(model, best_acc, processor, args)
    
if __name__ == "__main__":
    main()