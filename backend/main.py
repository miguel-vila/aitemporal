from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Atemporal Podcast RAG API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatMessage(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    message: str

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="Atemporal Podcast RAG API is running"
    )

@app.get("/")
async def root():
    return {"message": "Atemporal Podcast RAG API", "status": "running"}

@app.post("/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """Chat endpoint with enhanced stub responses for Atemporal podcast content"""
    try:
        user_message = message.message.lower().strip()
        
        # Enhanced contextual responses based on common Atemporal topics
        if any(word in user_message for word in ['venezuela', 'chávez', 'maduro', 'caracazo']):
            response_text = f"Sobre Venezuela y tu pregunta '{message.message}': Atemporal ha tenido conversaciones profundas sobre la situación venezolana con expertos como Juan Carlos Restrepo, Gabriela Febres-Cordero y Roberto Deniz. Estos episodios abordan desde el Caracazo hasta la actualidad, analizando las causas y consecuencias de la crisis."
            sources = [
                "Episodio #142 - Juan Carlos Restrepo: Chávez, Maduro, Saab, negacionismo y alarmismo",
                "Episodio #137 - Gabriela Febres-Cordero: El Caracazo, Chávez, Fidel",
                "Episodio #145 - Roberto Deniz: Venezuela, la codicia de Alex Saab"
            ]
        elif any(word in user_message for word in ['colombia', 'política', 'gobierno', 'estrategia']):
            response_text = f"Sobre Colombia y tu consulta '{message.message}': Atemporal ha explorado profundamente la realidad colombiana con líderes como Mauricio Cárdenas, Juan Carlos Echeverry y Alejandro Salazar. Las conversaciones abarcan desde la crisis del 98 hasta las estrategias para el futuro del país."
            sources = [
                "Episodio #190 - Alejandro Salazar: ¿Puede un país juicioso GANAR?",
                "Episodio #146 - Juan Carlos Echeverry: La crisis del 98, Colombia",
                "Episodio #133 - Mauricio Cárdenas: García Márquez, Don Guti, Plan Colombia"
            ]
        elif any(word in user_message for word in ['economía', 'finanzas', 'empresa', 'negocio']):
            response_text = f"Sobre economía y finanzas, relacionado con '{message.message}': Atemporal ha tenido fascinantes conversaciones sobre estrategia empresarial, crisis financieras y el futuro económico de la región con expertos como Sara Ordoñez y Alejandro Salazar."
            sources = [
                "Episodio #189 - Sara Ordoñez: A horas de un DESASTRE financiero",
                "Episodio #190 - Alejandro Salazar: ¿Puede un país juicioso GANAR?",
                "Serie de episodios sobre estrategia empresarial"
            ]
        elif any(word in user_message for word in ['temas', 'recurrentes', 'qué', 'cuáles']):
            response_text = "Los temas más recurrentes en Atemporal incluyen: la situación venezolana y sus lecciones para la región, el futuro de Colombia y sus desafíos institucionales, estrategias de desarrollo económico y empresarial, el rol de las élites en Latinoamérica, y reflexiones sobre liderazgo político."
            sources = [
                "Análisis transversal de 190+ episodios",
                "Conversaciones sobre Venezuela (múltiples episodios)",
                "Serie sobre liderazgo y gestión pública"
            ]
        else:
            # General responses for other topics
            import random
            responses = [
                f"¡Excelente pregunta sobre '{message.message}'! Atemporal es conocido por sus conversaciones profundas y sin prisa con personalidades excepcionales de Latinoamérica. Con más de 190 episodios, el podcast abarca temas cruciales como política, economía, estrategia y liderazgo.",
                f"Sobre tu consulta '{message.message}': El archivo de Atemporal incluye entrevistas con ex-presidentes, ministros, empresarios, académicos y pensadores de toda la región. Las conversaciones van desde análisis coyunturales hasta reflexiones estratégicas.",
                f"Interesante que preguntes sobre '{message.message}'. Andrés Acevedo ha construido un archivo único de conversaciones que capturan la complejidad de Latinoamérica. Cada episodio ofrece perspectivas profundas sobre los desafíos y oportunidades de la región."
            ]
            response_text = random.choice(responses)
            sources = [
                "Archivo completo de Atemporal (190+ episodios)",
                "Base de conversaciones con líderes latinoamericanos",
                "Transcripciones del podcast disponibles para análisis"
            ]
        
        return ChatResponse(
            response=response_text,
            sources=sources
        )
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor al procesar tu pregunta"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
