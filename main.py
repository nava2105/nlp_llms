import logging
from gettext import translation

import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

key = os.getenv('GOOGLE_API_KEY')

logging.basicConfig(
    filename="application.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

diary = """"Heredé de mis antepasados las ansias de huir. Dicen que mi sangre es europea. Yo siento que cada glóbulo 
procede de un punto distinto. De cada nación, de cada provincia, de cada isla, golfo, accidente, archipiélago, oasis. 
De cada trozo de tierra o de mar han usurpado algo y así me formaron, condenándome a la eterna búsqueda de un lugar de 
origen. Con los labios expresamente dibujados para exhalar quejas. Con la frente estrujada por todas las dudas. Con la 
malicia instintiva de la prohibición. Heredé el paso vacilante con objeto de no estatizarme nunca con firmeza en lugar 
alguno. ¡En todo y en nada! ¡En nada y en todo!"""

poems = [
    """
    Simplemente no soy de este mundo... Yo habito con frenesí la luna. No tengo miedo de morir; tengo miedo de esta 
    tierra ajena, agresiva... No puedo pensar en cosas concretas; no me interesan. Yo no sé hablar como todos. Mis 
    palabras son extrañas y vienen de lejos, de donde no es, de los encuentros con nadie... ¿Qué haré cuando me sumerja 
    en mis fantásticos sueños y no pueda ascender? Porque alguna vez va a tener que suceder. Me iré y no sabré volver. 
    Es más, nos sabré siquiera que hay un saber volver. No lo querré acaso.
    Alejandra Pizarnik
    """,

    """"
    La lenta máquina del desamor, los engranajes del reflujo, los cuerpos que abandonan las almohadas, las sábanas, 
    los besos, y de pie ante el espejo interrogándose cada uno a sí mismo, ya no mirándose entre ellos, ya no desnudos 
    para el otro, ya no te amo, mi amor. 
    
    Una poesía bastante clara que expresa cómo poco a poco se ha ido perdiendo la 
    magia y la ilusión en una relación de pareja, hasta el punto de haber desaparecido el amor.
    Julio Cortazar
    """,

    """"
    No es fácil expresar lo que has cambiado.
    Si ahora estoy viva entonces muerta he estado,
    aunque, como una piedra, sin saberlo,
    quieta en mi sitio, mi hábito siguiendo.
    No me moviste un ápice, tampoco
    me dejaste hacia el cielo alzar los ojos
    en paz, sin esperanza, por supuesto,
    de asir los astros o el azul con ellos.
    
    No fue eso. Dormí: una serpiente
    como una roca entre las rocas hiende
    el intervalo del invierno blanco,
    cual mis vecinos, nunca disfrutando
    del millón de mejillas cinceladas
    que a cada instante para fundir se alzan
    las mías de basalto. Como ángeles
    que lloran por la gente tonta hacen
    lágrimas que se congelan. Los muertos
    tenían yelmos helados. No les creo.
    
    Me dormí como un dedo curvo yace.
    Lo primero que vi fue puro aire
    y gotas que se alzaban de un rocío
    límpidas como espíritus. y miro
    densas y mudas piedras en tomo a mí,
    sin comprender. Reluzco y me deshojo
    como mica que a sí misma se escancie,
    igual que un líquido entre patas de ave,
    entre tallos de planta. Mas no pienses
    que me engañaste, eras transparente.
    
    Árbol y piedra nítidos, sin sombras.
    Mi dedo, cual cristal de luz sonora.
    Yo florecía como rama en marzo:
    una pierna y un brazo y otro brazo.
    De piedra a nube iba yo ascendiendo.
    A una especie de dios ya me asemejo,
    hiende el aire la veste de mi alma
    cual pura hoja de hielo. Es una dádiva.
    Sylvia Plath
    """,

    """"
    Pobre mi amor
    creíste
    que era así
    no supiste.
    Era más rico que eso
    era más pobre que eso
    era la vida y tú
    con los ojos cerrados
    viste tus pesadillas
    y dijiste
    la vida.
    Idea Vilariño
    """,

    """"
    No sabes como necesito tu voz;
    necesito tus miradas
    aquellas palabras que siempre me llenaban,
    necesito tu paz interior;
    necesito la luz de tus labios
    ¡¡¡ Ya no puedo... seguir así !!!
    ...Ya... No puedo
    mi mente no quiere pensar
    no puede pensar nada más que en ti.
    Necesito la flor de tus manos
    aquella paciencia de todos tus actos
    con aquella justicia que me inspiras
    para lo que siempre fue mi espina
    mi fuente de vida se ha secado
    con la fuerza del olvido...
    me estoy quemando;
    aquello que necesito ya lo he encontrado
    pero aun !!!Te sigo extrañando!!!
    Mario Benedetti
    """,

    """
    esta lúgubre manía de vivir
    esta recóndita humorada de vivir
    te arrastra alejandra no lo niegues
    
    hoy te miraste en el espejo
    y te fue triste estabas sola
    la luz rugía el aire cantaba
    pero tu amado no volvió
    
    enviarás mensajes sonreirás
    tremolarás tus manos así volverá
    tu amado tan amado
    
    oyes la demente sirena que lo robó
    el barco con barbas de espuma
    donde murieron las risas
    recuerdas el último abrazo
    oh nada de angustias
    ríe en el pañuelo llora a carcajadas
    pero cierra las puertas de tu rostro
    para que no digan luego
    que aquella mujer enamorada fuiste tú
    
    te remuerden los días
    te culpan las noches
    te duele la vida tanto tanto
    desesperada ¿adónde vas?
    desesperada ¡nada más!
    Alejandra Pizarnik
    """
]

def config_llm():
    try:
        genai.configure(api_key=key)
        logging.info("Google Generative AI configured successfully with provided API key.")
        model = genai.GenerativeModel("gemini-1.5-flash")
        logging.info("Model 'gemini-1.5-flash' initialized successfully.")
        return model
    except Exception as e:
        logging.error(f"Error configuring LLM: {e}")
        raise

def generate_embeddings():
    embeddings_with_texts = []
    try:
        for text in poems:
            result = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document",
                title="Text Embedding"
            )
            embeddings_with_texts.append((text, result['embedding']))
            logging.info("Embeddings generated successfully for all texts.")
        return embeddings_with_texts
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return None


def calculate_cosine_similarity(vector_a, vector_b):
    try:
        result = np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b))
        logging.info(f"Cosine similarity calculated: {result}")
        return result
    except Exception as e:
        logging.error(f"Error calculating cosine similarity: {e}")
        raise


def generate_response(model, embeddings_with_texts, question):
    try:
        question_embedding = genai.embed_content(
            model="models/embedding-001",
            content=question,
            task_type="retrieval_document",
            title="Question Embedding"
        )['embedding']

        similarities = [
            (calculate_cosine_similarity(question_embedding, embedding), text)
            for text, embedding in embeddings_with_texts
        ]

        sorted_texts = sorted(similarities, key=lambda x: x[0], reverse=True)

        threshold = 0.6
        top_texts = [text for similarity, text in sorted_texts if similarity > threshold]

        if not top_texts:
            logging.warning("No texts matched the threshold. Using the most similar text.")
            top_texts = [sorted_texts[0][1]]

        combined_context = "\n\n".join(top_texts)

        prompt = f"""Contexto:\n{combined_context}\n\nPregunta:\n{question}"""

        response = model.generate_content(prompt)
        logging.info("Response successfully generated by the model.")
        return response.text

    except Exception as e:
        Logging.error(f"Error generating response: {e}")
        return None

def generate_prompt (context, question):
    return f"""Contexto:\n{context}\n\nPregunta:\n{question}"""

def llm_test():
    model = config_llm()
    print("\nModel configured successfully.\n")
    print("NLP features:\n")
    print("1) Text Generation:\n")
    logging.info("---- Text Generation ----")
    question = "Explica como funciona la inteligencia artificial."
    print("Question for model: ", question)
    response = model.generate_content(question)
    print("Model answer: ", response.text)
    logging.info(f"Generated response for question '{question}': {response.text}")
    print("2) Automatic translation\n")
    logging.info("---- Automatic Translation ----")
    translation_prompt = generate_prompt(diary, "Traduce al ingles el texto (dame una sola respuesta): ")
    translation = model.generate_content(translation_prompt)
    print("Question for translation: ", translation_prompt)
    print("Model translation: ", translation.text)
    logging.info(f"Generated response for translation: {translation_prompt}': {translation.text}")
    print("3) Sentiment Analysis\n")
    logging.info("---- Sentiment Analysis ----")
    sentiment_analysis_prompt = generate_prompt(diary, "Analiza el sentimiento del siguiente texto y clasifícalo como positivo, negativo o neutral: ")
    sentiment_analysis = model.generate_content(sentiment_analysis_prompt)
    print("Question for sentiment analysis: ", sentiment_analysis_prompt)
    print("Model sentiment analysis: ", sentiment_analysis.text)
    logging.info(f"Generated response for sentiment analysis: {sentiment_analysis_prompt}': {sentiment_analysis.text}")
    print("4) Text Resume\n")
    logging.info("---- Text Resume ----")
    text_resume_prompt = generate_prompt(diary,
                                                "Resume el siguiente texto lo más pequeño posible: ")
    text_resume = model.generate_content(text_resume_prompt)
    print("Question for text resume: ", text_resume_prompt)
    print("Model text resume: ", text_resume.text)
    logging.info(
        f"Generated response for text resume: {text_resume_prompt}': {text_resume.text}")

    embeddings_with_texts = generate_embeddings()
    if embeddings_with_texts is not None:
        logging.info("Embeddings generated successfully.")
        print("\nEmbeddings generated successfully.\n")
        print("Text embeddings:")
        print(embeddings_with_texts)

        print("\nQuestion for embeddings:\n")
        question = "Cuales son los autores de cada uno de los poemas?"
        print(question,"\n")
        response = generate_response(model, embeddings_with_texts, question)
        print("Response:\n")
        print(response)
        logging.info(f"Answer for question '{question}': {response}")
        question = "Dame un análisis psicológico de esos autores basados en sus poemas y dime que trastornos puede tener cada uno de ellos."
        print(question, "\n")
        response = generate_response(model, embeddings_with_texts, question)
        print("Response:\n")
        print(response)
        logging.info(f"Answer for question '{question}': {response}")
    else:
        logging.error("Embeddings could not be generated.")

if __name__ == "__main__":
    logging.info("Program started.")
    try:
        logging.info(f"Google API Key: {key}")
        llm_test()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    finally:
        logging.info("Program finished execution.")