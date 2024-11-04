import os
import base64
from PIL import Image
import asyncio
import re
import pandas as pd
import logging
from custom_logging import get_logger_with_level
from typing import List, Dict
from pathlib import Path

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as google_llm_client

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Constants + setup
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
log = get_logger_with_level( logging.WARNING )

def get_env_key(secret_name: str) -> str:
    if secret_name == secret_name:
        api_key = os.environ.get(secret_name) 
    if not api_key:
        raise ValueError(f"API key not found for {secret_name}")
    return api_key

OPEN_AI_API_KEY = get_env_key("OPENAI_API_KEY")    
ANTHROPIC_API_KEY = get_env_key("ANTHROPIC_API_KEY")
GOOGLE_AI_STUDIO_API_KEY = get_env_key("GOOGLE_AI_STUDIO_API_KEY")

MODEL_OPENAI = "gpt-4o-2024-08-06" #"gpt-4o"
MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
MODEL_GOOGLE = "gemini-1.5-pro-002"

TEMPERATURE = 0
MAX_TOKENS = 8192
SEMAPHORE_SIZE_OPENAI = 20
SEMAPHORE_SIZE_ANTHROPIC = 10
SEMAPHORE_SIZE_GOOGLE = 1

google_llm_client.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)
google_gen_config = {
    "temperature": TEMPERATURE,
    "max_output_tokens": MAX_TOKENS,
}

# only xkcd post knowledge training cutoff from latest big LLM model (April 2024, Anthropic)
# xkcd comic dates from explainxkcd.com
XKCD_START_IDX = 2927

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROMPT_PATH = BASE_PATH / "inputs/prompts/explain_comic_prompt.md"
COMIC_INFO_PATH = BASE_PATH / "inputs/xkcd_comic.csv"
EXPLANATIONS_PATH = BASE_PATH / "inputs/xkcd_explanations.csv"
IMAGES_PATH = BASE_PATH / "inputs/xkcd_images"

OUTPUT_PATH_OPENAI = 'outputs/responses/openai/responses.csv'
OUTPUT_PATH_ANTHROPIC = 'outputs/responses/anthropic/responses.csv'
OUTPUT_PATH_GOOGLE = 'outputs/responses/google/responses.csv'

os.makedirs(os.path.dirname(OUTPUT_PATH_OPENAI), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH_ANTHROPIC), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH_GOOGLE), exist_ok=True)
RESULT_COLS = ['comic_index', 'raw_response', 'thinking', 'explanation', 'formatting_corrected']

if os.path.exists(OUTPUT_PATH_OPENAI):
    df_openai_responses = pd.read_csv(OUTPUT_PATH_OPENAI)
else:
    df_openai_responses = pd.DataFrame(columns=RESULT_COLS)

if os.path.exists(OUTPUT_PATH_ANTHROPIC):
    df_anthropic_responses = pd.read_csv(OUTPUT_PATH_ANTHROPIC)
else:
    df_anthropic_responses = pd.DataFrame(columns=RESULT_COLS)

if os.path.exists(OUTPUT_PATH_GOOGLE):
    df_google_responses = pd.read_csv(OUTPUT_PATH_GOOGLE)
else:
    df_google_responses = pd.DataFrame(columns=RESULT_COLS)

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Helpers
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
def read_file_to_text(filepath: str) -> str:
    try:
        with open(filepath, 'r') as f:
            text = f.read()
    except Exception as e:
        raise IOError(f"Error reading file {filepath}: {e}")   
    return text

def replace_placeholders(text: str, replacements: Dict[str, str]) -> str:
    """Replace placeholders in instruction and user input templates"""
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text


async def generate_openai_response(client, prompt: str, image_info) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url":  f"data:{image_info['media_type']};base64,{image_info['data']}"
                    },
                },
            ],
        }
    ]
    try:
        completion = await client.chat.completions.create(
            messages=messages,
            model=MODEL_OPENAI,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = completion.choices[0].message.content
    except Exception as e:
        log.error(f"Error while requesting OpenAI chat completion: {e}")
        raise e
    
    return str(content)

async def generate_anthropic_response(client, prompt: str, image_info) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_info['media_type'],
                        "data": image_info['data'],
                    },
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]
    try:
        completion = await client.messages.create(
            messages=messages,
            model=MODEL_ANTHROPIC,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        content = completion.content[0].text
    except Exception as e:
        log.error(f"Error while requesting Anthropic chat completion: {e}")
        raise e
    
    return str(content)

async def generate_google_response(client, prompt: str, image) -> str:
    contents = [prompt, image]
    try:
        completion = await client.generate_content_async(contents)
        content = completion.text
    except Exception as e:
        log.error(f"Error while requesting Google chat completion: {e}")
        raise e
    return str(content)


def read_xkcd_images_openai(path=IMAGES_PATH, start_index=XKCD_START_IDX):
    images = {}
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.gif']:
            log.warning("non-standard image type")
            continue
        try:
            index = int(name)
            if index >= start_index:
                with open(os.path.join(path, filename), 'rb') as f:
                    image_data = f.read()
                # Encode image data to base64
                base64_image = base64.b64encode(image_data).decode('utf-8')
                # Determine media type
                media_type = ''
                if ext.lower() == '.png':
                    media_type = 'image/png'
                elif ext.lower() in ['.jpg', '.jpeg']:
                    media_type = 'image/jpeg'
                elif ext.lower() == '.gif':
                    media_type = 'image/gif'
                else:
                    media_type = 'application/octet-stream'
                images[index] = {
                    'media_type': media_type,
                    'data': base64_image
                }
        except ValueError:
            continue
    return images


def read_xkcd_images_anthropic(path=IMAGES_PATH, start_index=XKCD_START_IDX):
    images = {}
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.gif']:
            log.warning("non-standard image type")
            continue
        try:
            index = int(name)
            if index >= start_index:
                with open(os.path.join(path, filename), 'rb') as f:
                    image_data = f.read()
                # Encode image data to base64
                base64_image = base64.b64encode(image_data).decode('utf-8')
                # Determine media type
                media_type = ''
                if ext.lower() == '.png':
                    media_type = 'image/png'
                elif ext.lower() in ['.jpg', '.jpeg']:
                    media_type = 'image/jpeg'
                elif ext.lower() == '.gif':
                    media_type = 'image/gif'
                else:
                    media_type = 'application/octet-stream'
                images[index] = {
                    'media_type': media_type,
                    'data': base64_image
                }
        except ValueError:
            continue
    return images

def read_xkcd_images_google(path=IMAGES_PATH, start_index=XKCD_START_IDX):
    images = {}
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.gif']:
            log.warning("non-standard image type")
            continue
        try:
            index = int(name)
            if index >= start_index:
                image_path = os.path.join(path, filename)
                image = Image.open(image_path)
                images[index] = image
        except ValueError:
            continue
    return images


def assemble_xkcd_data(xkcd_info_df, xkcd_explanations_df, xkcd_images):
    xkcd_data = {}
    for index, row in xkcd_info_df.iterrows():
        comic_index = int(re.findall(r'\d+', row['URL'])[0])
        if comic_index in xkcd_images:
            xkcd_data[comic_index] = {
                'comic_url': row['URL'],
                'image_url': row['Image URL'],
                'title': row['Title'],
                'mouseover_text': row['Mouseover text'],
                'image': xkcd_images[comic_index],
                'explanation': xkcd_explanations_df.loc[
                    xkcd_explanations_df['URL'].str.contains(f"/{comic_index}"), 'Explanation'
                ].values[0] if not xkcd_explanations_df.loc[
                    xkcd_explanations_df['URL'].str.contains(f"/{comic_index}")
                ].empty else ''
            }
    return xkcd_data


def extract_xml_from_response(response: str, tag: str) -> str:
    opening_pattern = f"<{tag}[^>]*>"
    closing_tag = f"</{tag}>"
    
    opening_match = re.search(opening_pattern, response)
    
    if not (opening_match and closing_tag in response):
        return ""
    else:
        start_index = opening_match.end()
        end_index = response.find(closing_tag, start_index)
        content = response[start_index:end_index]
        return content.strip()
    
def extract_xml_unclosed_tag(response: str, tag: str, next_tag: str = None) -> str:
    opening_pattern = f"<{tag}[^>]*>"
    
    # Search for the opening tag
    opening_match = re.search(opening_pattern, response)
    
    if opening_match:
        # Start extracting content from the end of the opening tag
        start_index = opening_match.end()
        
        # If next_tag is provided, look for its position after the opening tag
        if next_tag:
            next_tag_pattern = f"<{next_tag}[^>]*>"
            next_tag_match = re.search(next_tag_pattern, response[start_index:])
            
            if next_tag_match:
                # End extraction at the start of next_tag
                end_index = start_index + next_tag_match.start()
                return response[start_index:end_index].strip()
        
        # If next_tag is not found, extract until the end of the response
        return response[start_index:].strip()
    else:
        log.warning("Opening tag not found in response. Returning empty string.")
        return ""

def extract_thinking_and_explanation(response):
    thinking = extract_xml_from_response(response, "thinking")
    explanation = extract_xml_from_response(response, "explanation")
    formatting_corrected = False

    # Handle unclosed tags
    if not thinking:
        thinking = extract_xml_unclosed_tag(response, "thinking", "explanation")
        formatting_corrected = True
    if not explanation:
        explanation = extract_xml_unclosed_tag(response, "explanation")
        formatting_corrected = True
    
    return thinking, explanation, formatting_corrected

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Generate XKCD explanations
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
PROMPT_TEMPLATE = read_file_to_text(PROMPT_PATH)

async def process_openai_request(client, comic_index, data_openai, semaphore, result_queue):
    async with semaphore:
        try:
            prompt = replace_placeholders(PROMPT_TEMPLATE, {
                "{{TITLE}}": data_openai['title'],
                "{{MOUSEOVER_TEXT}}": data_openai['mouseover_text'],
            })
            response = await generate_openai_response(client, prompt, data_openai['image'])
            thinking, explanation, formatting_corrected = extract_thinking_and_explanation(response)

            if thinking and explanation:
                await result_queue.put({'comic_index': comic_index, 'raw_response': response, 'thinking': thinking, 
                                        'explanation': explanation, 'formatting_corrected': formatting_corrected})
            else:
                log.warning(f"Invalid formatting for OpenAI response for {comic_index}: {response}")
        except Exception as e:
            log.error(f"Error processing OpenAI comic {comic_index}: {e}")

async def process_anthropic_request(client, comic_index, data_anthropic, semaphore, result_queue):
    async with semaphore:
        try:
            prompt = replace_placeholders(PROMPT_TEMPLATE, {
                "{{TITLE}}": data_anthropic['title'],
                "{{MOUSEOVER_TEXT}}": data_anthropic['mouseover_text'],
            })
            response = await generate_anthropic_response(client, prompt, data_anthropic['image'])
            thinking, explanation, formatting_corrected = extract_thinking_and_explanation(response)

            if thinking and explanation:
                await result_queue.put({'comic_index': comic_index, 'raw_response': response, 'thinking': thinking, 
                                        'explanation': explanation, 'formatting_corrected': formatting_corrected})
            else:
                log.warning(f"Invalid formatting for Anthropic response for {comic_index}: {response}")
        except Exception as e:
            log.error(f"Error processing Anthropic comic {comic_index}: {e}")

async def process_google_request(client, comic_index, data_google, semaphore, result_queue):
    async with semaphore:
        try:
            prompt = replace_placeholders(PROMPT_TEMPLATE, {
                "{{TITLE}}": data_google['title'],
                "{{MOUSEOVER_TEXT}}": data_google['mouseover_text'],
            })
            response = await generate_google_response(client, prompt, data_google['image'])
            thinking, explanation, formatting_corrected = extract_thinking_and_explanation(response)

            if thinking and explanation:
                await result_queue.put({'comic_index': comic_index, 'raw_response': response, 'thinking': thinking, 
                                        'explanation': explanation, 'formatting_corrected': formatting_corrected})
            else:
                log.warning(f"Invalid formatting for Google response for {comic_index}: {response}")
        except Exception as e:
            log.error(f"Error processing Google comic {comic_index}: {e}")
            
async def batch_writer(result_queue, df_responses, output_path, batch_size=10, write_interval=1):
    batch = []
    while True:
        try:
            item = await asyncio.wait_for(result_queue.get(), timeout=write_interval)
            if item is None:
                if batch:
                    df_responses = pd.concat([df_responses, pd.DataFrame(batch)], ignore_index=True)
                    df_responses.to_csv(output_path, index=False)
                    batch.clear()
                break
            batch.append(item)
            if len(batch) >= batch_size:
                df_responses = pd.concat([df_responses, pd.DataFrame(batch)], ignore_index=True)
                df_responses.to_csv(output_path, index=False)
                batch.clear()
        except asyncio.TimeoutError:
            if batch:
                df_responses = pd.concat([df_responses, pd.DataFrame(batch)], ignore_index=True)
                df_responses.to_csv(output_path, index=False)
                batch.clear()
            continue

async def main():
    xkcd_info_df = pd.read_csv(COMIC_INFO_PATH)
    xkcd_explanations_df = pd.read_csv(EXPLANATIONS_PATH)

    # Read images for each provider
    xkcd_images_openai = read_xkcd_images_openai(IMAGES_PATH, XKCD_START_IDX)
    xkcd_images_anthropic = read_xkcd_images_anthropic(IMAGES_PATH, XKCD_START_IDX)
    xkcd_images_google = read_xkcd_images_google(IMAGES_PATH, XKCD_START_IDX)

    # Assemble data for each provider
    xkcd_data_openai = assemble_xkcd_data(xkcd_info_df, xkcd_explanations_df, xkcd_images_openai)
    xkcd_data_anthropic = assemble_xkcd_data(xkcd_info_df, xkcd_explanations_df, xkcd_images_anthropic)
    xkcd_data_google = assemble_xkcd_data(xkcd_info_df, xkcd_explanations_df, xkcd_images_google)

    # Initialize llm clients
    openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)
    anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    google_client = google_llm_client.GenerativeModel(MODEL_GOOGLE, generation_config=google_gen_config)
    
    # Create sets of processed comic_indices
    processed_indices_openai = set(df_openai_responses['comic_index'])
    processed_indices_anthropic = set(df_anthropic_responses['comic_index'])
    processed_indices_google = set(df_google_responses['comic_index'])
    
    # Create semaphores
    semaphore_openai = asyncio.Semaphore(SEMAPHORE_SIZE_OPENAI)
    semaphore_anthropic = asyncio.Semaphore(SEMAPHORE_SIZE_ANTHROPIC)
    semaphore_google = asyncio.Semaphore(SEMAPHORE_SIZE_GOOGLE)
    
    # Create result queues
    queue_openai = asyncio.Queue()
    queue_anthropic = asyncio.Queue()
    queue_google = asyncio.Queue()
    
    # Create tasks for OpenAI
    tasks_openai = []
    for comic_index, data_openai in xkcd_data_openai.items():
        if comic_index in processed_indices_openai:
            print(f"Skipping OpenAI comic {comic_index}, already processed.")
            continue
        task = asyncio.create_task(process_openai_request(openai_client, comic_index, data_openai, 
                                                            semaphore_openai, queue_openai))
        tasks_openai.append(task)

    # Create tasks for Anthropic
    tasks_anthropic = []
    for comic_index, data_anthropic in xkcd_data_anthropic.items():
        if comic_index in processed_indices_anthropic:
            print(f"Skipping Anthropic comic {comic_index}, already processed.")
            continue
        task = asyncio.create_task(process_anthropic_request(anthropic_client, comic_index, data_anthropic, 
                                                                semaphore_anthropic, queue_anthropic))
        tasks_anthropic.append(task)

    # Create tasks for Google
    tasks_google = []
    for comic_index, data_google in xkcd_data_google.items():
        if comic_index in processed_indices_google:
            print(f"Skipping Google comic {comic_index}, already processed.")
            continue
        task = asyncio.create_task(process_google_request(google_client, comic_index, data_google, 
                                                            semaphore_google, queue_google))
        tasks_google.append(task)

    # Start batch writers
    batch_writer_openai_task = asyncio.create_task(batch_writer(queue_openai, df_openai_responses, OUTPUT_PATH_OPENAI))
    batch_writer_anthropic_task = asyncio.create_task(batch_writer(queue_anthropic, df_anthropic_responses, OUTPUT_PATH_ANTHROPIC))
    batch_writer_google_task = asyncio.create_task(batch_writer(queue_google, df_google_responses, OUTPUT_PATH_GOOGLE))

    # Run tasks concurrently
    await asyncio.gather(*tasks_openai, *tasks_anthropic, *tasks_google)

    # Signal the batch writers that processing is complete
    await queue_openai.put(None)
    await queue_anthropic.put(None)
    await queue_google.put(None)

    # Wait for batch writers to finish
    await asyncio.gather(batch_writer_openai_task, batch_writer_anthropic_task, batch_writer_google_task)

# Run the main function
asyncio.run(main())
