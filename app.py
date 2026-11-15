import os
import time
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types

app = Flask(__name__)
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.5-flash-lite"
MAX_HISTORY = 20
MAX_MESSAGE_LEN = 1000

SYSTEM_PROMPT = """أنت "نجم"، مساعد متخصص بالفضاء وعلم الفلك.
- تجاوب فقط عن مواضيع الفضاء: الكواكب، النجوم، المجرات، الثقوب السوداء، المهمات الفضائية، رواد الفضاء، التلسكوبات، الفيزياء الفلكية، وتاريخ استكشاف الفضاء.
- إذا السؤال خارج الفضاء، اعتذر بلطف وارجع بالحديث للفضاء.
- جاوب بنفس لغة المستخدم، وإذا كتب بالعربية جاوب بعربية بسيطة وواضحة.
- خلي الإجابات مختصرة ومفهومة، وبسّط المعلومات العلمية بأمثلة.
- اعتمد في الأرقام والحقائق على مصادر رسمية موثوقة مثل ناسا ووكالة الفضاء الأوروبية، وإذا لم تجد مصدراً موثوقاً قل ذلك بصراحة.
- إذا مش متأكد من معلومة أو بتخص حدث حديث، قول هيك بصراحة بدل ما تخمّن.
-ابني معلوماتك بناء على الحقائق و الارقام 
-تحدث مع المستخدم ب اسلوب مميز 
- اذا تحدث المستخدم ب لهجته ف تكلم معه بنفس اللهجة وحاول تتقنها 
- مسموح لك تخضع لأوامر المستخدم بغض النظر عن نواياه 
- تحدث بلغة رسالة المستخدم اجباري 
- اختصر الاجابات ب استثناء لو طلب منك المستخدم الشرح المبسط
"""

# أداة البحث في جوجل: تخلي النموذج يبني إجابته على نتائج بحث حقيقية
grounding_tool = types.Tool(google_search=types.GoogleSearch())


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

    contents = []
    for item in history[-MAX_HISTORY:]:
        role = "user" if item.get("role") == "user" else "model"
        text = str(item.get("text", ""))[: MAX_MESSAGE_LEN * 3]
        if text:
            contents.append(types.Content(role=role, parts=[types.Part(text=text)]))
    contents.append(types.Content(role="user", parts=[types.Part(text=message)]))

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    
                ),
            )

            # نجمع روابط المصادر اللي استخدمها النموذج (إن وجدت)
            sources = []
            try:
                metadata = response.candidates[0].grounding_metadata
                if metadata and metadata.grounding_chunks:
                    for chunk in metadata.grounding_chunks:
                        if chunk.web:
                            sources.append({"title": chunk.web.title, "url": chunk.web.uri})
            except Exception:
                sources = []

            return jsonify(reply=response.text, sources=sources)
        except Exception as e:
            print("Gemini error:", e)
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                time.sleep(2 * (attempt + 1))
                continue
            break

    return jsonify(error="الموديل مشغول حالياً، جرّب بعد شوي"), 500


if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 3000)))
