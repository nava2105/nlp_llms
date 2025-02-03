# nlp_llms

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Google Gemini](https://img.shields.io/badge/google%20gemini-8E75B2?style=for-the-badge&logo=google%20gemini&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Tabla de Contenidos
1. [Información General](#informacion-general)
2. [Tecnologías](#tecnologias)
3. [Instalación](#instalacion)
4. [Uso](#uso)
5. [Estructura del Proyecto](#estructura-del-proyecto)
6. [Notas](#notas)

## Información General
***
Este proyecto es un **Chatbot basado en IA generativa** que permite:
- Configurar y utilizar modelos de lenguaje de Google Generative AI.
- Generar respuestas basadas en embeddings semánticos.
- Analizar sentimientos, traducir texto y generar resúmenes.
- Implementar una interfaz interactiva basada en línea de comandos.
- Registrar logs de actividad para depuración.

## Tecnologías
***
Tecnologías utilizadas en el proyecto:
- [Python](https://www.python.org): Versión 3.12.0
- [Google Generative AI](https://ai.google.dev/): Para generación de respuestas y embeddings.
- [NumPy](https://numpy.org/): Para cálculos matemáticos como similitud de coseno.
- [dotenv](https://pypi.org/project/python-dotenv/): Para manejo de variables de entorno.
- [Logging](https://docs.python.org/3/library/logging.html): Para registrar la actividad del sistema.

## Instalación
***
### Requisitos Previos
Verifica que tienes Python 3.12.0 instalado:
```bash
python --version
```

### Instalación del Proyecto

#### Clonar el Repositorio
Descarga los archivos del proyecto:
```bash
git clone git clone https://github.com/nava2105/nlp_llms.git
cd chatbot_ai
```

#### Crear y Activar un Entorno Virtual
Se recomienda usar un entorno virtual para administrar las dependencias:
```bash
python -m venv .venv
# Activar en Windows
.venv\Scripts\activate
# Activar en macOS/Linux
source .venv/bin/activate
```

#### Instalar Dependencias
Instala las librerías necesarias desde `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Configuración del Entorno
Crea un archivo `.env` en el directorio raíz y agrega tu clave de API de Google:
```env
GOOGLE_API_KEY=tu_clave_aqui
```

### Ejecución de la Aplicación
Después de la configuración, ejecuta el cualquiera de los dos archivos:
```bash
python main.py
```
```bash
python chatbots.py
```

## Uso
***
### Funcionalidades Clave
1. **Generación de Texto:**
   - Genera respuestas basadas en modelos de IA de Google.
2. **Traducción Automática:**
   - Traduce texto a diferentes idiomas.
3. **Análisis de Sentimiento:**
   - Identifica el tono de un texto (positivo, negativo o neutral).
4. **Embeddings Semánticos:**
   - Utiliza modelos de embeddings para mejorar la comprensión del contexto.
5. **Chatbot Interactivo:**
   - Conversa con la IA a través de línea de comandos.

### Cómo Usarlo
1. **Inicia la Aplicación:**
   ```bash
   python main.py
   ```
    ```bash
    python chatbots.py
    ```
2. **Selecciona el Modo de Chatbot:**
   - `1` para Chatbot API.
   - `2` para Chatbot basado en embeddings.
3. **Escribe tus preguntas y obtén respuestas generadas por IA.**
4. **Para salir, escribe `quit`.**

## Estructura del Proyecto
***
```plaintext
chatbot_ai/
├── main.py               # Código principal para inicializar el chatbot.
├── chatbots.py           # Funciones para generar respuestas y embeddings.
├── requirements.txt      # Dependencias del proyecto.
├── .env                  # Variables de entorno (API Key de Google AI).
└── README.md             # Documentación del proyecto.
```

## Notas
***
- **Claves de API:** Asegúrate de obtener una clave API válida de [Google Generative AI](https://ai.google.dev/).
- **Gestión de Errores:** Los logs se guardan en `application.log` y `chatbots_logs.log` para depuración.
- **Uso de Embeddings:** Para mejorar respuestas, el chatbot utiliza embeddings de textos predefinidos.

Estos códigos proporciona una base sólida para interactuar con modelos de IA generativa y puede ampliarse con nuevas funcionalidades en el futuro.

