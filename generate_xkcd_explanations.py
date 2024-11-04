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

MODEL_OPENAI = "gpt-4o"
MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
MODEL_GOOGLE = "gemini-1.5-pro-002"

openai_client = AsyncOpenAI(api_key=OPEN_AI_API_KEY)
anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
google_llm_client.configure(api_key=GOOGLE_AI_STUDIO_API_KEY)
google_llm = google_llm_client.GenerativeModel(MODEL_GOOGLE)

TEMPERATURE = 0
MAX_TOKENS = 2048
SEMAPHORE_SIZE_OPENAI = 20
SEMAPHORE_SIZE_ANTHROPIC = 10
SEMAPHORE_SIZE_GOOGLE = 1

# only xkcd post knowledge training cutoff from latest big LLM model (April 2024, Anthropic)
# xkcd comic dates from explainxkcd.com
XKCD_START_IDX = 2927

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
PROMPT_PATH = BASE_PATH / "inputs/prompts/explain_comic_prompt.md"
COMIC_INFO_PATH = BASE_PATH / "inputs/xkcd_comic.csv"
EXPLANATIONS_PATH = BASE_PATH / "inputs/xkcd_explanations.csv"
IMAGES_PATH = BASE_PATH / "inputs/xkcd_images"

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


async def generate_openai_response(prompt: str, image_info) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image",
                    "image": {
                        "media_type": image_info['media_type'],
                        "data": image_info['data'],
                    },
                },
            ],
        }
    ]
    try:
        completion = await openai_client.chat.completions.create(
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

async def generate_anthropic_response(prompt: str, image_info) -> str:
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
        completion = await anthropic_client.messages.create(
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

async def generate_google_response(prompt: str, image) -> str:
    contents = [prompt, image]
    try:
        completion = await google_llm.generate_content_async(
            contents,
            generation_config=google_llm.types.GenerationConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            )
        )
        content = completion.text
    except Exception as e:
        log.error(f"Error while requesting Google chat completion: {e}")
        raise e
    return str(content)


def read_xkcd_images(path=IMAGES_PATH, start_index=XKCD_START_IDX):
    images = {}
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.gif']:
            continue
        try:
            index = int(name)
            if index >= start_index:
                with open(os.path.join(path, filename), 'rb') as f:
                    image_data = f.read()
                images[index] = image_data
        except ValueError:
            continue
    return images

def read_xkcd_images_openai(path=IMAGES_PATH, start_index=XKCD_START_IDX):
    images = {}
    for filename in os.listdir(path):
        name, ext = os.path.splitext(filename)
        if ext.lower() not in ['.png', '.jpg', '.jpeg', '.gif']:
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


def extract_xml(response: str, tag: str):
    opening_pattern = f"<{tag}[^>]*>"
    closing_tag = f"</{tag}>"
    
    opening_match = re.search(opening_pattern, response)
    
    if not (opening_match and closing_tag in response):
        log.warning("Unable to extract information for given tag (LLM likely did not follow response "
                    "formatting instructions). Returning full response")
        return response
    else:
        start_index = opening_match.end()
        end_index = response.find(closing_tag, start_index)
        content = response[start_index:end_index]
        return content.strip()

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Generate XKCD explanations
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
PROMPT_TEMPLATE = read_file_to_text(PROMPT_PATH)

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
