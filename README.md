
# Enhanced Visual Question Answering for Preschool Education with ViLT, BLIP-2, and Sinkhorn Distance

This project is an innovative Visual Question Answering (VQA) system designed specifically for preschool children, addressing tasks like counting, logic, shape recognition, and color identification. Our solution integrates the ViLT-VQAv2 and BLIP-2 models to generate accurate answers based on both image and text inputs.

<img src="Resources\transformer-vqa-model.jpg" alt="alt text" title="Sample image caption">

## Vision-and-Language Transformer (ViLT), fine-tuned on VQAv2

Vision-and-Language Transformer (ViLT) model fine-tuned on [VQAv2](https://visualqa.org/). It was introduced in the paper [ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](https://arxiv.org/abs/2102.03334) by Kim et al. and first released in [this repository](https://github.com/dandelin/ViLT). 


## Bootstrapping Language-Image Pre-training (BLIP-2)

Bootstrapping Language-Image Pre-training (BLIP-2) model is designed for generating high-quality image captions and aligning vision-language representations. It was introduced in the paper [BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models](https://arxiv.org/abs/2301.12597) by Li et al. and was first released in [this repository](https://github.com/salesforce/BLIP).

Disclaimer: The team releasing ViLT and Blip-2 did not write a model card for this model so this model card has been written by the Hugging Face team.

## Key Innovations

- **Fusion of ViLT-VQAv2 and BLIP-2**: Combines ViLT's robust visual and language understanding with BLIP-2’s ability to generate high-quality image captions, improving VQA results.
  
- **Image Captioning**: BLIP-2 generates captions from the input images, which are then used as additional context for the VQA model.
  
- **Sinkhorn Distance**: Incorporates Sinkhorn distance to align features and improve accuracy in counting and logic-based tasks.

- **Consistency Classifier**: Introduces a consistency classifier to ensure that the answers are logical and consistent with both the image and question context.


## Intended uses & limitations

You can use the raw model for visual question answering. 

### How to use

Here is how to use this model in PyTorch:

```python
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = "How many cats are there?"

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

# prepare inputs
encoding = processor(image, text, return_tensors="pt")

# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()
print("Predicted answer:", model.config.id2label[idx])
```

