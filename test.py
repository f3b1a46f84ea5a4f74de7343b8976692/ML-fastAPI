from transformers import MarianMTModel, MarianTokenizer

model_name = 'Helsinki-NLP/opus-mt-ar-ru'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

arabic_text = "مرحبا كيف حالك؟"
inputs = tokenizer(arabic_text, return_tensors="pt", padding=True)
translated = model.generate(**inputs)
russian_text = tokenizer.decode(translated[0], skip_special_tokens=True)

print(russian_text)
