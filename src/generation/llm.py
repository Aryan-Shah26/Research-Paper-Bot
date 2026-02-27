from huggingface_hub import InferenceClient
from src.generation.prompts import SYSTEM_PROMPT, get_messages

def load_llm(token: str) -> InferenceClient:
    return InferenceClient(token=token)

def generate_answer(client: InferenceClient, retrieved_docs: list, question: str) -> dict:
    #Build the context string from the retrieved documents
    
    if not retrieved_docs :
        return {
            "answer" : "I could not find the relevant information in the given documents.",
            "sources" : []
        }
    
    try :
        context = "\n\n".join([
            f"[{doc.metadata['source']} - Page {doc.metadata['page']}]: {doc.page_content}" for doc in retrieved_docs
        ])

        #Format the system prompt with the context and question
        prompt = SYSTEM_PROMPT.format(context=context, question=question)

        #Call the LLM to generate an answer
        response = client.chat_completion(
        model="meta-llama/Llama-3.2-3B-Instruct",  # ✅ swap this
        messages=get_messages(context, question),
        max_tokens=1024,
        temperature=0.2,
    )

        return{
            "answer" : response.choices[0].message.content.strip(),
            "sources" : [f"{doc.metadata['source']} - Page {doc.metadata['page']}" for doc in retrieved_docs]
        }
    
    except HfHubHTTPError as e :
        if "429" in str(e) :
            raise RuntimeError("API rate limits reached. Please try again later.")
        elif "401" in str(e) :
            raise RuntimeError("Invalid Hugging Face token. Please check your .env file")
        elif "400" in str(e) :
            raise RuntimeError("Model request failed. The model may be unavailable or the input may be too long.")
        else :
            raise RuntimeError(f"API error : {str(e)}")
        
    except Exception as e :
        raise RuntimeError(f"Unexpected error during generation: {str(e)}")
