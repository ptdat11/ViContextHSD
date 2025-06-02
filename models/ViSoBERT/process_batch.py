from transformers import AutoTokenizer, ViTImageProcessor

if "processor" not in globals():
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
if "tokenizer" not in globals():
    tokenizer = AutoTokenizer.from_pretrained('5CD-AI/visobert-14gb-corpus')

def process_batch(batch: dict):
    if "caption" in batch:
        batch["caption"] = tokenizer.batch_encode_plus(
            batch["caption"], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        )
    if "image" in batch:
        batch["image"] = processor(
            batch["image"],
            return_tensors="pt"
        )
    if "comment" in batch:
        batch["comment"] = tokenizer.batch_encode_plus(
            batch["comment"], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        )
    if "reply" in batch:
        batch["reply"] = tokenizer.batch_encode_plus(
            batch["reply"], 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        )
    
    return batch