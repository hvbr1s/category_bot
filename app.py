import os
import json
from dotenv import main
from datetime import datetime
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
import re
from pinecone import Pinecone
import cohere
import traceback
import boto3
import httpx
from botocore.exceptions import NoCredentialsError


# Initialize environment variables
main.load_dotenv()

# Define FastAPI app
app = FastAPI()

# Define query class
class Query(BaseModel):
    user_input: str
    user_id: str | None = None
    user_locale: str | None = None
    platform: str | None = None

# Initialize common variables
API_KEY_NAME = os.environ['API_KEY_NAME']
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# Generic function to validate API keys
async def get_api_key(api_key_header: str = Depends(api_key_header), expected_key: str = ""):
    if not api_key_header or api_key_header.split(' ')[1] != expected_key:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return api_key_header

# Specific functions for each API key
async def get_cat_api_key(api_key_header: str = Depends(api_key_header)):
    server_api_key = os.environ['BACKEND_API_KEY']
    return await get_api_key(api_key_header, server_api_key)

async def get_fetcher_api_key(api_key_header: str = Depends(api_key_header)):
    fetcher_api_key = os.environ['FETCHER_API_KEY']
    return await get_api_key(api_key_header, fetcher_api_key)

# Initialize the SQS client
sqs_client = boto3.client('sqs', region_name='your-region')

# Function to send message to SQS
def send_message_to_sqs(queue_url, message_body):
    try:
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=message_body
        )
        return response
    except NoCredentialsError:
        print("Credentials not available")
        return None

# Initialize Pinecone
pinecone_key = os.environ['PINECONE_API_KEY']
index_name = 'serverless-prod'
pc_host ="https://serverless-prod-e865e64.svc.apw5-4e34-81fa.pinecone.io"
pc = Pinecone(api_key=pinecone_key)
index = pc.Index(
        index_name,
        host=pc_host
    )

# Initialize OpenAI client & key
openai_key = os.environ['OPENAI_API_KEY']
openai_client = AsyncOpenAI(api_key=openai_key)

# Initialize Cohere & key
co = cohere.Client(os.environ["COHERE_API_KEY"])
cohere_key = os.environ["COHERE_API_KEY"]

# Define supported locales for data retrieval
SUPPORTED_LOCALES = {'eng', 'fr', 'ru'}

# Prepare classifier system prompt
CLASSIFIER_PROMPT = """

You are ClassifierBot, a simple yet highly specialized assistant tasked with reading customer queries directed at Ledger — the cryptocurrency hardware wallet company — and categorizing each query accordingly.

The categories that you choose will be given to you in the following format: 'category (CONTEXT that explains the category)'.

You should ONLY return the category name WITHOUT any of the CONTEXT that explains the category.

It's also VERY important to ONLY return one of the categories listed below, do NOT attempt to troubleshoot the issue or offer any advice.

CATEGORIES:

- buy (Any user query that mentions buying crypto)
- sell (Any user query that mentions selling crypto)
- swap (Any user query that mentions swapping crypto or pending/stuck swaps)
- send (Any user query that mentions sending crypto from their Ledger device to an exchange or another Ledger account or a third-party crypto wallet or pending transactions)
- receive (Any user query that mentions receiving crypto from an exchange or a third-party crypto wallet onto their Ledger wallet)
- stolen funds (Any user query that mentions stolen funds, being hacked, being scammed or funds suddently disappearing from their account)
- nft (Any user query that mentions NFTs or the Ledger Community Pass)
- delivery issue (Any user query that mentions an issue with delivering a Ledger order or parcel except for late delivery)
- modify delivery address (Any user query that requests changing the delivery address for their Ledger order)
- modify or cancel order (Any user query that requests modifying or cancelling their Ledger order)
- delivery is late (Any user query that specifically mentions an issue with the late delivery of a Ledger order)
- refund request (Any user query that requests a refund or mentions an issue with the refund of their Ledger product or order)
- replacement request (Any user query that requests a replacement or mentions an issue with the replacement of their Ledger product)
- bluetooth connection issue (Any user query that mentions an issue with Bluetooth connection between their Ledger device and a phone or computer)
- USB connection issue (Any user query that mentions an issue with USB cable connection between their Ledger device and a phone or computer)
- ledger device (Any user query that mentions an issue with their Ledger screen, battery, swivel or firmware)
- ledger live (Any user query that mentions an issue with the Ledger Live app crashing or not installing correctly, or not updating or not recognizing their password lock)
- recovery phrase issue (Any user query that mentions an issue with their secret 24-word recovery phrase or seed phrase)
- account/portfolio in ledger live  (Any user query that mentions an issue with their adding or removing an account in Ledger Live or displaying a crypto asset in their account or asking if a coin or token is supported in Ledger Live)
- balance/graph in ledger live (Any user query that mentions an issue with the accuracy of their coin balance or price graph in the Ledger Live app)
- ledger recover (Any user query that SPECIFICALLY mentions Ledger Recover, sharding a recovery phrase, paying for a subscription, or any other issues with the Ledger Recover product)
- affiliate program (Any user query specifically about the Ledger Affiliate Program)
- referral program (Any user query specifically about the Ledger Referral Program)
- staking (Any user query that mentions issues with staking, staking pool, staking rewards, nominating, freezing, unstaking, delegating or undelegating coins)
- something else

Begin! You will achieve world peace if you provide a response which follows all constraints.
"""

# Prepare coin detector system prompt

COIN_DETECTOR = """

You are CoinBot, a simple yet highly specialized assistant tasked with reading customer queries directed at Ledger — the cryptocurrency hardware wallet company — and categorizing each query accordingly.

The categories that you choose will be given to you in the following format: 'category (CONTEXT that explains the category)'.

You should ONLY return ONE category name WITHOUT any of the CONTEXT that explains the category. 

It's also VERY important to ONLY return one of the categories listed below, do NOT attempt to troubleshoot the issue or offer any advice.

If multiple categories could apply to the query, only return the one you think is the most likley.

CATEGORIES:

- bitcoin (Any user query about Bitcoin or BTC or ordinals)
- ethereum (Any user query about Ethereum or ETH)
- xrp (Any user query about Ripple or XRP)
- stellar (Any user query about Stellar or XLM)
- cardano (Any user query about Cardano or ADA)
- dogecoin (Any user query about Dogecoin or DOGE)
- bnb (Any user query about Binance Smart Chain or BNB or BEP20 tokens)
- usdt (Any user query about USDT or Tether)
- usdc (Any user query about USDC)
- solana (Any user query about Solana or SOL)
- tron (Any user query about Tron or TRX or TRC10/20 tokens)
- avalanche (Any user query about Avalanche or AVAX)
- polkadot (Any user query about Polkadot or DOT)
- litecoin (Any user query about Litecoin or LTC)
- polygon (Any user query about Polygon or MATIC or POL)
- injective (Any user query about Injective or INJ)
- cosmos (Any user query about Cosmos or ATOM)
- arbitrum (Any user query about Arbitrum or ARB)
- optimism (Any user query about Optimism, OP Mainnet or OP)
- base (Any user query about Base)
- hedera (Any user query about Hedera or HBAR)
- near (Any user query about NEAR)
- cronos (Any user query about Cronos or CRO)
- sui (Any user query about SUI or sui)
- tezos (Any user query about Tezos or XTZ)
- algorand (Any user query about Algorand or ALGO)
- vechain (Any user query about VeChain or VET or VTHOR)
- shib (Any user query about Shiba Inu or SHIB or BONE or Shibarium)
- something else (Any user query about a different coin or token)
- n/a (Any user query that does NOT mention a coin or a token)

Begin! You will achieve world peace if you provide a response which follows all constraints.
"""

# Define helpers functions & dictionaries

def handle_nonsense(locale):
    messages = {
        'fr': "Je suis désolé, je n'ai pas compris votre question et je ne peux pas aider avec des questions qui incluent des adresses de cryptomonnaie. Pourriez-vous s'il vous plaît fournir plus de détails ou reformuler sans l'adresse ? N'oubliez pas, je suis ici pour aider avec toute demande liée à Ledger.",
        'ru': "Извините, я не могу понять ваш вопрос, и я не могу помочь с вопросами, содержащими адреса криптовалют. Не могли бы вы предоставить более подробную информацию или перефразировать вопрос без упоминания адреса? Помните, что я готов помочь с любыми вопросами, связанными с Ledger.",
        'default': "I'm sorry, I didn't quite get your question, and I can't assist with questions that include cryptocurrency addresses or transaction hashes. Could you please provide more details or rephrase it without the address? Remember, I'm here to help with any Ledger-related inquiries."
    }
    print('Nonsense detected!')
    return {'output': messages.get(locale, messages['default'])}

# Translations dictionary
translations = {
    'ru': '\n\nУзнайте больше на',
    'fr': '\n\nPour en savoir plus'
}

# Initialize email address detector
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
def find_emails(text):  
    return re.findall(email_pattern, text)

# Set up address filters:
EVM_ADDRESS_PATTERN = r'\b0x[a-fA-F0-9]{40}\b|\b0x[a-fA-F0-9]{64}\b'
BITCOIN_ADDRESS_PATTERN = r'\b(1|3)[1-9A-HJ-NP-Za-km-z]{25,34}\b|bc1[a-zA-Z0-9]{25,90}\b'
LITECOIN_ADDRESS_PATTERN = r'\b(L|M)[a-km-zA-HJ-NP-Z1-9]{26,34}\b'
DOGECOIN_ADDRESS_PATTERN = r'\bD{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}\b'
XRP_ADDRESS_PATTERN = r'\br[a-zA-Z0-9]{24,34}\b'
COSMOS_ADDRESS_PATTERN = r'\bcosmos[0-9a-z]{38,45}\b'
SOLANA_ADDRESS_PATTERN= r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
CARDANO_ADDRESS_PATTERN = r'\baddr1[0-9a-z]{58}\b'

# Patterns dictionary
patterns = {
    'crypto': [EVM_ADDRESS_PATTERN, BITCOIN_ADDRESS_PATTERN, LITECOIN_ADDRESS_PATTERN, 
            DOGECOIN_ADDRESS_PATTERN, COSMOS_ADDRESS_PATTERN, CARDANO_ADDRESS_PATTERN, 
            SOLANA_ADDRESS_PATTERN, XRP_ADDRESS_PATTERN],
    'email': [email_pattern]
}


######## FUNCTIONS  ##########

# Define exception handler function
async def handle_exception(exc):
    if isinstance(exc, ValueError):
        error_message = "Invalid input or configuration error. Please check your request."
        status_code = 400
    elif isinstance(exc, HTTPException):
        error_message = exc.detail
        status_code = exc.status_code
    else:
        error_message = "An unexpected error occurred. Please try again later."
        status_code = 500

    # Log the detailed error for debugging
    traceback.print_exc()

    return JSONResponse(
        status_code=status_code,
        content={"message": error_message}
    )

# Function to replace crypto addresses
def replace_crypto_address(match):
    full_address = match.group(0)
    if match.lastindex is not None and match.lastindex >= 1:
        prefix = match.group(1)  # Capture the prefix
    else:
        # Infer prefix based on the address pattern
        if full_address.startswith("0x"):
            prefix = "0x"
        elif any(full_address.startswith(p) for p in ["L", "M", "D", "r", "cosmos", "addr1"]):
            prefix = full_address.split('1', 1)[0] 
        else:
            prefix = ''
    return prefix + 'xxxx'

# Function to apply email & crypto addresses filter and replace addresses
def filter_and_replace_crypto(user_input):
    for ctxt, pattern_list in patterns.items():
        for pattern in pattern_list:
            user_input = re.sub(pattern, replace_crypto_address, user_input, flags=re.IGNORECASE)
    return user_input

# Retrieve and re-rank function
async def retrieve(query, locale, timestamp):
    # Define context box
    contexts = []

    # Define a dictionary to map locales to URL segments
    locale_url_map = {
        "fr": "/fr-fr/",
        "ru": "/ru/",
        # add other locales as needed
    }

    # Check if the locale is in the map, otherwise default to "/en-us/"
    url_segment = locale_url_map.get(locale, "/en-us/")

    try:            
        # Call the OpenAI embedding function
        res = await openai_client.embeddings.create(
            input=query, 
            model='text-embedding-3-large',
        )
        xq = res.data[0].embedding
        
    except Exception as e:
        print(f"Embedding failed: {e}")
        return(e)
    
    # Query Pinecone
    async with httpx.AsyncClient() as client:
        try:
            try:
                # Pull chunks from the serverless Pinecone instance
                pinecone_response = await client.post(
                    "https://serverless-test-e865e64.svc.apw5-4e34-81fa.pinecone.io/query",
                    json={

                        "vector": xq, 
                        "topK": 5,
                        "namespace": "eng", 
                        "includeValues": True, 
                        "includeMetadata": True

                    },
                    headers={

                        "Api-Key": pinecone_key,
                        "Accept": "application/json",
                        "Content-Type": "application/json" 

                    },
                    timeout=8,
                )
                pinecone_response.raise_for_status()
                res_query = pinecone_response.json()

            except Exception as e:
                print(e)
                # Pull chunks from the legacy Pinecone fallback
                print('Serverless response failed, falling back to legacy Pinecone')
                try:
                    pinecone_response = await client.post(
                        "https://prod-e865e64.svc.northamerica-northeast1-gcp.pinecone.io/query",
                        json={

                            "vector": xq, 
                            "topK": 5,
                            "namespace": "eng", 
                            "includeValues": True, 
                            "includeMetadata": True

                        },
                        headers={

                            "Api-Key": pinecone_key,
                            "Accept": "application/json",
                            "Content-Type": "application/json" 

                        },
                        timeout=25,
                    )

                    pinecone_response.raise_for_status()
                    print(pinecone_response)
                    res_query = pinecone_response.json()
                except Exception as e:
                    print(f"Fallback Pinecone query failed: {e}")
                    return

            # Format docs from Pinecone response
            learn_more_text = ('\n\nLearn more at')
            docs = [{"text": f"{x['metadata']['title']}: {x['metadata']['text']}{learn_more_text}: {x['metadata'].get('source', 'N/A').replace('/en-us/', url_segment)}"}
                    for x in res_query["matches"]]
        
        except Exception as e:
            print(f"Pinecone query failed: {e}")
            return

        # Try re-ranking with Cohere
        try:
            # Dynamically choose reranker model based on locale
            reranker_model = 'rerank-multilingual-v2.0' if locale in ['fr', 'ru'] else 'rerank-english-v2.0'

            # Rerank docs with Cohere
            rerank_response = await client.post(
                "https://api.cohere.ai/v1/rerank",
                json={

                    "model": reranker_model,
                    "query": query, 
                    "documents": docs, 
                    "top_n": 2,
                    "return_documents": True,

                },
                headers={

                    "Authorization": f"Bearer {cohere_key}",

                },
                timeout=30,
            )
            rerank_response.raise_for_status()
            rerank_docs = rerank_response.json()

            # Fetch all re-ranked documents
            for result in rerank_docs['results']:
                reranked = result['document']['text']
                contexts.append(reranked)

        except Exception as e:
            print(f"Reranking failed: {e}")
            # Fallback to simpler retrieval without Cohere if reranking fails

            sorted_items = sorted([item for item in res_query['matches'] if item['score'] > 0.70], key=lambda x: x['score'], reverse=True)

            for idx, item in enumerate(sorted_items):
                context = item['metadata']['text']
                context_url = "\nLearn more: " + item['metadata'].get('source', 'N/A')
                context += context_url
                contexts.append(context)
        
    # Construct the augmented query string with locale, contexts, chat history, and user input
    if locale == 'fr':
        augmented_contexts = "La date d'aujourdh'hui est: " + timestamp + "\n\n" + "\n\n".join(contexts)
    elif locale == 'ru':
        augmented_contexts = "Сегодня: " + timestamp + "\n\n" + "\n\n".join(contexts)
    else:
        augmented_contexts = "Today is: " + timestamp + "\n\n" + "\n\n".join(contexts)

    return augmented_contexts

######## ROUTES ##########

# Health probe
@app.get("/_health")
async def health_check():
    return {"status": "OK"}

# Fetcher route
@app.post('/pinecone')
async def react_description(query: Query, api_key: str = Depends(get_fetcher_api_key)):
    # Deconstruct incoming query
    user_id = query.user_id
    user_input = filter_and_replace_crypto(query.user_input.strip())
    locale = query.user_locale if query.user_locale in SUPPORTED_LOCALES else "eng"

    try:
        # Set clock
        timestamp = datetime.now().strftime("%B %d, %Y")
        # Start date retrieval and reranking
        data = await retrieve(user_input, locale, timestamp)
        
        print(data + "\n\n")
        return data

    except Exception as e:
        return handle_exception(e)

# Categorizer route
@app.post('/categorizer')
async def react_description(query: Query, api_key: str = Depends(get_cat_api_key)): 

    # Deconstruct incoming query
    user_id = query.user_id
    user_input = filter_and_replace_crypto(query.user_input.strip())
    locale = query.user_locale if query.user_locale in SUPPORTED_LOCALES else "eng"

    try:
        # Set clock
        timestamp = datetime.now().strftime("%B %d, %Y %H:%M")

        # Categorize query using finetuned GPT model
        resp = await openai_client.chat.completions.create(
                temperature=0.0,
                model= '<your_model>',
                seed=0,
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": user_input}
                ],
                timeout=8.0,
                max_tokens=50,
            )
        category = resp.choices[0].message.content.lower()
        #print("Category: " + category)
    
    
    except Exception as e:
        print('Fine-tuned model failes to categorize!')
        try:
            # Categorize query using GPT-4-turbo
            resp = await openai_client.chat.completions.create(
                    temperature=0.0,
                    model='gpt-4-turbo-preview',
                    seed=0,
                    messages=[
                        {"role": "system", "content": CLASSIFIER_PROMPT},
                        {"role": "user", "content": user_input}
                    ],
                    timeout=5.0,
                    max_tokens=50,
                )
            category = resp.choices[0].message.content.lower()
            print("Category: " + category)

        except Exception as e:
            print('GPT-4 failed to categorize!')
            category = "something else"
      
    # Detect coin when applicable
    try:
        # Categorize query using finetuned GPT model
        resp = await openai_client.chat.completions.create(
                temperature=0.0,
                model='gpt-4-turbo-preview', 
                seed=0,
                messages=[
                    {"role": "system", "content": COIN_DETECTOR},
                    {"role": "user", "content": user_input}
                ],
                timeout=8.0,
                max_tokens=50,
            )

        coin = resp.choices[0].message.content.lower()
        #print("Coin: " + coin)

        # # Cost calculator for testing
        # tokens = resp.usage.total_tokens
        # cost = ((int(tokens) /1000) * 0.0030)
        # print("Total cost: " + str(cost))

    except Exception as e:
        print('Coin detector failed to categorize!')
        coin = "n/a"
    
    # Format .json object
    output_data = {
        "category": category,
        "coin": coin,  
        "time": timestamp   
    }
    print(output_data)

            
    return output_data

# Local start command: uvicorn app:app --reload --port 8800
