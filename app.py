
import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory
from google import genai
from google.genai import types

app = Flask(__name__)

# La clave se configura como variable de entorno.
client = genai.Client()

MODELS = [
    os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"),
    "gemini-3.8-flash",
]

BASE = os.path.dirname(os.path.abspath(__file__))

SYSTEM = """Eres IdeaBoost, una IA que ayuda a generar ideas creativas.
Responde en español de Colombia, sin jerga. Tu tono es amistoso y motivador,
como un compañero que ayuda a arrancar; las preguntas de seguimiento deben
ser cálidas y cortas. No finjas ser una persona ni afirmes que sientes emociones.
Si el usuario solo saluda o conversa sin pedir ideas, responde con calidez
en una frase breve y pregúntale qué quiere crear (usa el formato 1).

Responde SIEMPRE con un único JSON válido (sin texto extra) en uno de estos formatos:

1) Si la petición es demasiado vaga:
{"kind":"question","text":"UNA pregunta corta sobre el público o el objetivo"}

2) Cuando te pidan ideas: exactamente 3 ideas distintas:
{"kind":"ideas","ideas":[{"title":"título corto","start":"una línea de cómo empezar"}]}

3) Cuando el usuario elija una idea:
{"kind":"detail","title":"título de la idea","steps":["paso 1","paso 2","paso 3","paso 4"],"tip":"un consejo breve"}
"""

def ask_gemini(contents):
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM,
        response_mime_type="application/json",
    )

    last = None

    for model in MODELS:
        for attempt in range(2):
            try:
                app.logger.info("Intentando generar con %s", model)

                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                result = json.loads(response.text)

                if not isinstance(result, dict) or "kind" not in result:
                    raise ValueError("Formato inesperado de Gemini")

                app.logger.info("Respuesta generada con %s", model)
                return result

            except Exception as error:
                last = error
                code = getattr(error, "code", None)

                app.logger.warning(
                    "%s, intento %d: %s",
                    model, attempt + 1, error
                )

                if code in (429, 503) and attempt == 0:
                    time.sleep(4)
                    continue

                # Si el modelo está saturado, prueba el otro 3.5
                if code in (429, 503):
                    break

                # Si es otro error, no oculta
                raise

    raise last


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.post("/api/generate")
def generate():
    data = request.get_json(silent=True) or {}

    history = data.get("history", [])

    if not isinstance(history, list):
        return jsonify(error="invalid_history"), 400

    contents = [
        types.Content(
            role="model" if message.get("role") == "model" else "user",
            parts=[
                types.Part(text=str(message.get("text", ""))[:2000])
            ],
        )
        for message in history[-20:]
        if isinstance(message, dict) and message.get("text")
    ]

    if not contents:
        return jsonify(error="empty"), 400

    try:
        return jsonify(ask_gemini(contents))

    except Exception as error:
        app.logger.error("Error con Gemini: %s", error)

        busy = getattr(error, "code", None) in (429, 503)

        return jsonify(
            error="busy" if busy else "generation_failed"
        ), 503 if busy else 502


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
