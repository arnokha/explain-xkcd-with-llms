# explain-xkcd-with-llms

## Github page for viewing explanations
[View LLM explanations for XKCDs](https://arnokha.github.io/explain-xkcd-with-llms/view_llm_xkcd_explanations.html)

# Running the scripts

## Getting the images
Refer to the [download repo](https://github.com/arnokha/download-xkcd-and-explanations) for how to get `inputs/images`, or you can amend `generate_xkcd_explanations.py` image reading functions to use the image URL from the `inputs/xkcd_comics.csv`.

## Installing Requirements
(not sure if I missed anything)
```bash
pip install -r requirements.txt
```

## Setting API Keys
```bash
export OPENAI_API_KEY='your-openai-api-key'
export ANTHROPIC_API_KEY='your-anthropic-api-key'
export GOOGLE_AI_STUDIO_API_KEY='your-google-api-key'
```

## Generate explanations
This script asynchronously generates explanations for XKCD comics using OpenAI, Anthropic, and Google language models. Set the semaphore size (e.g. `SEMAPHORE_SIZE_OPENAI`) based on your usage limits. Refer to `inputs/prompts/explain_comic_prompt.md` for prompt.
```bash
python generate_xkcd_explanations.py
```

## Judge explanations
This script uses Claude 3.5 Sonnet to judge all of the explanations. Modify semaphore size as needed. Refer to `inputs/prompts/judge_explanation.md` for prompt.
```bash
python python judge_explanations.py
```