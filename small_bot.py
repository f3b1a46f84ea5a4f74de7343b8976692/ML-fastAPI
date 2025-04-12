import logging
import asyncio
from aiogram import Bot, Dispatcher, types
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from langdetect import detect
from deep_translator import GoogleTranslator
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

logging.basicConfig(level=logging.INFO)

TOKEN = "7031250350:AAGFGTw65urjqe6Pvi_HLq7RN6J7qJePgmo"

bot = Bot(token=TOKEN)
dp = Dispatcher()

model_name = "AnatoliiPotapov/T-lite-instruct-0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def translate(text, target_language):
    translator = GoogleTranslator(target=target_language)
    return translator.translate(text)

def detect_language(text):
    lang = detect(text)
    if lang in ['ar', 'tr', 'en']:
        return lang
    return 'en'

def generate_response(text):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@dp.message()
async def handle_message(message: types.Message):
    user_text = message.text
    source_language = detect_language(user_text)
    translated_to_ru = translate(user_text, 'ru')
    ai_response_ru = generate_response(translated_to_ru)
    translated_back = translate(ai_response_ru, source_language)
    await message.reply(translated_back)

if __name__ == '__main__':
    dp.start_polling(bot)
    