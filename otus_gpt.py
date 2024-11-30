import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset
import gradio as gr

MODEL_NAME = "gpt2"
OUTPUT_DIR = "./fine_tuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

data_json = [
    {"text": "Пример диалога: Привет! Как дела? У меня всё отлично, спасибо! А у тебя?"},
    {"text": "Пример диалога: Что нового в жизни? Ничего особенного, всё по-старому. А у тебя?"},
    {"text": "Пример диалога: Какое сегодня число? Сегодня 30 ноября, если не ошибаюсь."},
    {"text": "Пример диалога: Какой твой любимый фильм? Мне нравятся фантастические фильмы, особенно «Начало»."},
    {"text": "Пример диалога: Ты знаешь шутки? Конечно! Вот одна: Почему программисты любят зиму? Потому что в декабре 31 день!"},
    {"text": "Пример диалога: Как научиться говорить на другом языке? Практикуйся каждый день, слушай носителей языка и пробуй разговаривать на нём как можно чаще."},
    {"text": "Пример диалога: Расскажи анекдот. Вот анекдот: Почему программистам нельзя доверить чайник? Потому что они ждут, пока 'зависнет'!"}
]

if not os.path.exists("data.json"):
    import json

    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(data_json, f, ensure_ascii=False, indent=4)

data = load_dataset("json", data_files="data.json")

if not os.path.exists(OUTPUT_DIR):
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="steps",
        save_steps=500,
        logging_dir="./logs",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        num_train_epochs=3,
        save_total_limit=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data["train"]
    )

    print("Начало дообучения модели...")
    trainer.train()
    print("Донастройка завершена.")

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
else:
    print("Модель уже дообучена. Загружаем сохранённую модель...")
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)


def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Введите запрос..."),
    outputs="text",
    title="GPT-like чат-бот",
    description="Введите текст, чтобы получить ответ от дообученной GPT-like модели."
)

if __name__ == "__main__":
    interface.launch()
