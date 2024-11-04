import os
import asyncio
import re
import pandas as pd
import logging
from custom_logging import get_logger_with_level
from typing import Dict
from pathlib import Path

from anthropic import AsyncAnthropic

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Constants + setup
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
log = get_logger_with_level(logging.WARNING)

def get_env_key(secret_name: str) -> str:
    api_key = os.environ.get(secret_name)
    if not api_key:
        raise ValueError(f"API key not found for {secret_name}")
    return api_key

ANTHROPIC_API_KEY = get_env_key("ANTHROPIC_API_KEY")

MODEL_JUDGE = "claude-3-5-sonnet-20241022"
SEMAPHORE_SIZE = 5
TEMPERATURE = 0
MAX_TOKENS = 8192

BASE_PATH = Path(os.path.dirname(os.path.realpath(__file__)))
JUDGING_PROMPT_PATH = BASE_PATH / "inputs/prompts/judge_explanation.md"
EXPLANATIONS_PATH = BASE_PATH / "inputs/xkcd_explanations.csv"
COMIC_INFO_PATH = BASE_PATH / "inputs/xkcd_comic.csv"
RESPONSES_PATH_OPENAI = BASE_PATH / 'outputs/openai/responses.csv'
RESPONSES_PATH_ANTHROPIC = BASE_PATH / 'outputs/anthropic/responses.csv'
RESPONSES_PATH_GOOGLE = BASE_PATH / 'outputs/google/responses.csv'

OUTPUT_DIR = BASE_PATH / 'outputs/judging'
OUTPUT_PATH_OPENAI = OUTPUT_DIR / 'openai/scores.csv'
OUTPUT_PATH_ANTHROPIC = OUTPUT_DIR / 'anthropic/scores.csv'
OUTPUT_PATH_GOOGLE = OUTPUT_DIR / 'google/scores.csv'

os.makedirs(OUTPUT_PATH_OPENAI.parent, exist_ok=True)
os.makedirs(OUTPUT_PATH_ANTHROPIC.parent, exist_ok=True)
os.makedirs(OUTPUT_PATH_GOOGLE.parent, exist_ok=True)

RESULT_COLS = [
    'comic_index',
    'explanation_model',
    'judge_model',
    'judge_response_raw',
    'judge_response_discussion',
    'judge_response_percentage'
]

if os.path.exists(OUTPUT_PATH_OPENAI):
    df_scores_openai = pd.read_csv(OUTPUT_PATH_OPENAI)
else:
    df_scores_openai = pd.DataFrame(columns=RESULT_COLS)

if os.path.exists(OUTPUT_PATH_ANTHROPIC):
    df_scores_anthropic = pd.read_csv(OUTPUT_PATH_ANTHROPIC)
else:
    df_scores_anthropic = pd.DataFrame(columns=RESULT_COLS)

if os.path.exists(OUTPUT_PATH_GOOGLE):
    df_scores_google = pd.read_csv(OUTPUT_PATH_GOOGLE)
else:
    df_scores_google = pd.DataFrame(columns=RESULT_COLS)

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
    """Replace placeholders in the template."""
    for key, value in replacements.items():
        text = text.replace(key, value)
    return text

def extract_judge_response_sections(response: str) -> Dict[str, str]:
    discussion = extract_xml_from_response(response, 'discussion')
    percentage_covered = extract_xml_from_response(response, 'percentage_covered')

    if not discussion:
        discussion = extract_xml_unclosed_tag(response, 'discussion', 'percentage_covered')
    if not percentage_covered:
        percentage_covered = extract_xml_unclosed_tag(response, 'percentage_covered')

    return {
        'judge_response_raw': response,
        'judge_response_discussion': discussion,
        'judge_response_percentage': percentage_covered.strip()
    }

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

    opening_match = re.search(opening_pattern, response)

    if opening_match:
        start_index = opening_match.end()

        if next_tag:
            next_tag_pattern = f"<{next_tag}[^>]*>"
            next_tag_match = re.search(next_tag_pattern, response[start_index:])

            if next_tag_match:
                end_index = start_index + next_tag_match.start()
                return response[start_index:end_index].strip()

        return response[start_index:].strip()
    else:
        log.warning("Opening tag not found in response. Returning empty string.")
        return ""

def extract_comic_index_from_url(url):
    match = re.search(r'/(\d+)/', url)
    if match:
        return int(match.group(1))
    else:
        # Try another pattern if necessary
        match = re.search(r'(\d+)', url)
        if match:
            return int(match.group(1))
    return None

##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
## Judging
##-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Read the judging prompt template
JUDGING_PROMPT_TEMPLATE = read_file_to_text(JUDGING_PROMPT_PATH)

async def judge_candidate_explanation(client, comic_index, explanation_model, data, semaphore, result_queue):
    async with semaphore:
        try:
            prompt = replace_placeholders(JUDGING_PROMPT_TEMPLATE, {
                "{{TITLE}}": data['title'],
                "{{MOUSEOVER_TEXT}}": data['mouseover_text'],
                "{{EXPLAIN_XKCD_EXPLANATION}}": data['ground_truth'],
                "{{CANDIDATE_EXPLANATION}}": data['candidate_explanation'],
            })

            response = await generate_anthropic_judgement_response(client, prompt)
            sections = extract_judge_response_sections(response)

            if sections['judge_response_discussion'] and sections['judge_response_percentage']:
                await result_queue.put({
                    'comic_index': comic_index,
                    'explanation_model': explanation_model,
                    'judge_model': MODEL_JUDGE,
                    'judge_response_raw': sections['judge_response_raw'],
                    'judge_response_discussion': sections['judge_response_discussion'],
                    'judge_response_percentage': sections['judge_response_percentage']
                })
            else:
                log.warning(f"Invalid formatting for judge response for comic {comic_index}, exp model {explanation_model}: {response}")

        except Exception as e:
            log.error(f"Error processing judgment for comic {comic_index}: {e}")

async def generate_anthropic_judgement_response(client, prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        completion = await client.messages.create(
            messages=messages,
            model=MODEL_JUDGE,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.content[0].text
    except Exception as e:
        log.error(f"Error while requesting judgment completion: {e}")
        raise e

async def batch_writer(result_queue, df_scores, output_path, batch_size=10, write_interval=1):
    batch = []
    while True:
        try:
            item = await asyncio.wait_for(result_queue.get(), timeout=write_interval)
            if item is None:
                if batch:
                    df_scores = pd.concat([df_scores, pd.DataFrame(batch)], ignore_index=True)
                    df_scores.to_csv(output_path, index=False)
                    batch.clear()
                break
            batch.append(item)
            if len(batch) >= batch_size:
                df_scores = pd.concat([df_scores, pd.DataFrame(batch)], ignore_index=True)
                df_scores.to_csv(output_path, index=False)
                batch.clear()
        except asyncio.TimeoutError:
            if batch:
                df_scores = pd.concat([df_scores, pd.DataFrame(batch)], ignore_index=True)
                df_scores.to_csv(output_path, index=False)
                batch.clear()
            continue


async def main():
     # Read necessary data
    xkcd_explanations_df = pd.read_csv(EXPLANATIONS_PATH)
    xkcd_comic_df = pd.read_csv(COMIC_INFO_PATH)

    # Extract comic_index from URLs
    xkcd_explanations_df['comic_index'] = xkcd_explanations_df['URL'].apply(extract_comic_index_from_url)
    xkcd_comic_df['comic_index'] = xkcd_comic_df['URL'].apply(extract_comic_index_from_url)

    # Merge the DataFrames on comic_index
    merged_df = pd.merge(xkcd_explanations_df, xkcd_comic_df, on='comic_index', how='inner')

    # Create ground_truth_map
    ground_truth_map = {}
    for _, row in merged_df.iterrows():
        comic_index = row['comic_index']
        ground_truth_map[comic_index] = {
            'ground_truth': row['Explanation'],
            'title': row['Title'],  # Adjust as needed based on column names
            'mouseover_text': row['Mouseover text']
        }
    
    # Read candidate explanations
    df_responses_openai = pd.read_csv(RESPONSES_PATH_OPENAI)
    df_responses_anthropic = pd.read_csv(RESPONSES_PATH_ANTHROPIC)
    df_responses_google = pd.read_csv(RESPONSES_PATH_GOOGLE)

    # Create data structures
    candidate_explanations = []

    for _, row in df_responses_openai.iterrows():
        candidate_explanations.append({
            'comic_index': row['comic_index'],
            'explanation_model': 'gpt-4o-2024-08-06',
            'candidate_explanation': row['explanation']
        })

    for _, row in df_responses_anthropic.iterrows():
        candidate_explanations.append({
            'comic_index': row['comic_index'],
            'explanation_model': 'claude-3-5-sonnet-20241022',
            'candidate_explanation': row['explanation']
        })

    for _, row in df_responses_google.iterrows():
        candidate_explanations.append({
            'comic_index': row['comic_index'],
            'explanation_model': 'gemini-1.5-pro-002',
            'candidate_explanation': row['explanation']
        })

    # Filter candidate explanations to those with ground truth
    candidate_explanations = [
        ce for ce in candidate_explanations if ce['comic_index'] in ground_truth_map
    ]

    # Initialize Anthropics client
    anthropic_client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Create semaphore
    semaphore = asyncio.Semaphore(SEMAPHORE_SIZE)

    # Create result queues
    queue_openai = asyncio.Queue()
    queue_anthropic = asyncio.Queue()
    queue_google = asyncio.Queue()

    # Create tasks
    tasks = []
    for ce in candidate_explanations:
        data = ground_truth_map[ce['comic_index']]
        data.update({'candidate_explanation': ce['candidate_explanation']})

        if "gpt-4o" in ce['explanation_model']:
            result_queue = queue_openai
        elif "claude" in ce['explanation_model']:
            result_queue = queue_anthropic
        elif "gemini" in ce['explanation_model']:
            result_queue = queue_google
        else:
            log.warning("unknown explanation model")
            continue

        task = asyncio.create_task(judge_candidate_explanation(
            anthropic_client,
            ce['comic_index'],
            ce['explanation_model'],
            data,
            semaphore,
            result_queue
        ))
        tasks.append(task)

    # Start batch writers
    batch_writer_openai_task = asyncio.create_task(batch_writer(
        queue_openai, df_scores_openai, OUTPUT_PATH_OPENAI))
    batch_writer_anthropic_task = asyncio.create_task(batch_writer(
        queue_anthropic, df_scores_anthropic, OUTPUT_PATH_ANTHROPIC))
    batch_writer_google_task = asyncio.create_task(batch_writer(
        queue_google, df_scores_google, OUTPUT_PATH_GOOGLE))

    # Run tasks concurrently
    await asyncio.gather(*tasks)

    # Signal the batch writers that processing is complete
    await queue_openai.put(None)
    await queue_anthropic.put(None)
    await queue_google.put(None)

    # Wait for batch writers to finish
    await asyncio.gather(
        batch_writer_openai_task,
        batch_writer_anthropic_task,
        batch_writer_google_task
    )


# Run the main function
asyncio.run(main())
