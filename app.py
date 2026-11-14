import os
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

app = Flask(__name__)

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-2.5-flash"  # غيّره إذا في موديل أحدث متاح عندك

MAX_HISTORY = 20       # أقصى عدد رسائل سابقة بنرسلها للموديل
MAX_MESSAGE_LEN = 1000  # أقصى طول للرسالة الواحدة

SYSTEM_PROMPT = """أنت "نجم"، مساعد متخصص بالفضاء وعلم الفلك.
- تجاوب فقط عن مواضيع الفضاء: الكواكب، النجوم، المجرات، الثقوب السوداء، المهمات الفضائية، رواد الفضاء، التلسكوبات، الفيزياء الفلكية، وتاريخ استكشاف الفضاء.
- إذا السؤال خارج الفضاء، اعتذر بلطف وارجع بالحديث للفضاء.
- جاوب بنفس لغة المستخدم، وإذا كتب بالعربية جاوب بعربية بسيطة وواضحة.
- خلي الإجابات مختصرة ومفهومة، وبسّط المعلومات العلمية بأمثلة.
- إذا مش متأكد من معلومة أو بتخص حدث حديث، قول هيك بصراحة بدل ما تخمّن.
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not message:
        return jsonify(error="اكتب رسالة أولاً"), 400
    if len(message) > MAX_MESSAGE_LEN:
        return jsonify(error="الرسالة طويلة، اختصرها شوي"), 400

    # نبني سجل المحادثة بالشكل اللي بيفهمه Gemini
    contents = []
    for item in history[-MAX_HISTORY:]:
        role = "user" if item.get("role") == "user" else "model"
        text = str(item.get("text", ""))[:MAX_MESSAGE_LEN * 3]
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT),
        )
        return jsonify(reply=response.text)
    except Exception as e:
        print("Gemini error:", e)
        return jsonify(error="صار خطأ بالاتصال بالذكاء الاصطناعي، جرّب بعد شوي"), 500


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 3000)))
