# Simple Chatbot (FastAPI + Ollama + Docker)

A simple, local, general-purpose chatbot built as a hands-on project to learn how LLM inference, APIs, and containerization fit together. No cloud costs — runs entirely on your own machine using a local open-source model.

## What it does

Send a text message to the API, get a reply back from a locally running LLM (Llama 3.2 via Ollama). No training or fine-tuning involved — this is pure inference (using a pre-trained model as-is).

## Tech Stack

- **Python** — core language
- **FastAPI** — web framework serving the API
- **Ollama** — runs the Llama 3.2 model locally and exposes it over HTTP
- **Docker** — packages the app so it runs consistently anywhere

## How it works

1. Ollama runs in the background on the host machine, serving Llama 3.2 at `http://localhost:11434`
2. The FastAPI app exposes a `POST /chat` endpoint
3. When a message comes in, FastAPI forwards it to Ollama's `/api/chat` endpoint
4. Ollama returns the model's reply as JSON
5. FastAPI extracts the reply text and sends it back to the caller

```

## Project Structure

```
.
├── main.py              # FastAPI app with the /chat endpoint
├── test_ollama.py       # Scratch file used to test raw calls to Ollama
├── requirements.txt     # Python dependencies
├── Dockerfile           # Instructions to build the container image
├── .dockerignore        # Files excluded from the Docker build
└── README.md
```

## Running with Docker

1. Make sure Ollama is running on the host machine (outside the container)
2. Build the image:
   ```bash
   docker build -t simple-chatbot .
   ```
3. Run the container:
   ```bash
   docker run -p 8000:8000 simple-chatbot
   ```
4. Test it:
   ```bash
   curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"message": "Tell me a fun fact."}'
   ```

**Note:** Inside the container, the app talks to Ollama via `http://host.docker.internal:11434` instead of `localhost`, since `localhost` inside a container refers to the container itself, not the host machine.

## Example Request/Response

**Request**
```json
POST /chat
{
  "message":"Tell me a fun fact."
}
```

**Response**
```json
{
  "reply":"Octopuses have three hearts."
}
```
## Requirements

Listed in `requirements.txt`.
