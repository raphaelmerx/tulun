# Tulun

[![license](https://img.shields.io/badge/License-MIT-blue)](https://github.com/raphaelmerx/tulun/blob/main/LICENSE)
![versions](https://img.shields.io/badge/python-3.12-blue.svg)

Transparent and Adaptable Low-resource Machine Translation, through LLM post-editing

[🎥 Demo video](https://youtu.be/fQFwOxzR4MI) | [🖥️ Live demo (using Bislama)](https://bislama-trans.rapha.dev/) | [📄 Paper](https://arxiv.org/abs/2505.18683)

![Tulun Demo](./demo.gif)

## Local installation

1. Install Python dependencies:
```bash
python -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
```

2. Setup credentials: create a .env file in the root directory and add the following:
```bash
# for Google Translate (optional)
GOOGLE_APPLICATION_CREDENTIALS='<credential-file>.json'
# for Gemini, can also use OpenAI / Anthropic / others, see https://docs.litellm.ai/docs/
GEMINI_API_KEY='<api-key>'
```

3. Run the server:
```bash
./manage.py migrate && ./manage.py runserver
```

You can now configure your install (target lang, import glossary, etc.) at http://localhost:8000/admin/. After that, you can access the translation interface at http://localhost:8000/.

## Deployment

1. Install Docker and Docker Compose

2. Setup credentials: create a prod.env file in the root directory and add credentials, similar to example above.

3. Run the server:
```bash
docker-compose up -d
```

Access your server at http://localhost:8008/. Can be deployed behind a reverse proxy like Nginx.

## Evaluation

For evaluations in the paper: see the [eval](./eval/) folder README.

For the in-app eval mode, upload your evaluation set at `/admin/translations/evalrow/`. Upon entering a sentence that is part of the eval set, the app will automatically switch to eval mode.
