import asyncio
import json
import logging
import random
import signal
from typing import List

import aiofiles
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm

# Configuration
INPUT_FILE = "secba_process.jsonl"
OUTPUT_FILE = "secba_processgpt.jsonl"
PROMPT_FILE = "zpromptsgpt.jsonl"
STARTING_FILE = "startingfilegpt.jsonl"

MAX_TOKENS = 115000
MAX_RETRIES = 50
INITIAL_RETRY_DELAY = 1  # seconds
STARTING_POINT = 0
NO_WORKERS = 5

# Ignore SIGHUP (hangup signal)
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# Load environment variables from the specified path
load_dotenv("/home/arthrod/.env/.env")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("contract_processor_detailed.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# Initialize AsyncOpenAI client
client = AsyncOpenAI()


class ContractExtraction(BaseModel):
    document_name: str
    document_type: str | None
    agreement_name: str | None
    parties: List[str]
    agreement_date: str | None
    effective_date: str | None
    expiration_date: str | None
    contract_amount: float | None
    currency: str | None
    payment_terms: str | None
    governing_law: str | None
    jurisdiction: str | None
    signatories: List[str] | None
    amendment_history: List[str] | None
    contract_term_period: str | None
    renewal_terms: str | None
    summary: str
    operative_clause: str
    confidentiality: str | None
    termination: str | None
    indemnification: str | None
    limitation_of_liability: str | None
    intellectual_property: str | None
    dispute_resolution: str | None
    force_majeure: str | None
    assignment: str | None
    non_compete: str | None
    non_solicitation: str | None
    warranties: str | None
    insurance: str | None
    audit_rights: str | None
    data_protection: str | None
    compliance_with_laws: str | None
    affiliate_license_licensee: str | None
    anti_assignment: str | None
    change_of_control: str | None
    claims: str | None
    competitive_restriction: str | None
    covenant_not_to_sue: str | None
    early_termination: str | None
    engagement: str | None
    entire_agreement: str | None
    escrow: str | None
    exclusivity: str | None
    fees: str | None
    ip_ownership: str | None
    license_grant: str | None
    liquidated_damages: str | None
    minimum_commitment: str | None
    payment_and_fees: str | None
    price_restrictions: str | None
    renewal_term: str | None
    representations_and_warranties: str | None
    scope_of_use: str | None
    services: str | None
    severability_clause: str | None
    survival: str | None
    taxes: str | None
    term: str | None
    termination_for_convenience: str | None
    third_party_beneficiary: str | None
    waiver: str | None
    average_confidence: float
    total_sections: int


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("API call timed out")


# Set the signal handler and a 10-minute alarm
signal.signal(signal.SIGALRM, timeout_handler)


def adjust_prompt_length(prompt, max_tokens=115000):
    # Get the encoding
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    # Count the number of tokens in the prompt
    num_tokens = len(encoding.encode(prompt))

    # If the number of tokens exceeds the maximum, reduce the prompt size
    if num_tokens > max_tokens:
        logger.warning(
            f"Prompt exceeds maximum length of {max_tokens} tokens. Truncating..."
        )

        # Calculate the percentage to keep
        keep_ratio = max_tokens / num_tokens

        # Find the split point
        split_point = int(len(prompt) * keep_ratio)

        # Split the prompt
        truncated_content = prompt[:split_point]

        # Add the end of instructions
        end_of_instructions = """

</document>

2. Analyze the document text thoroughly. Pay attention to key details such as names, dates, amounts, all clauses, all aliases, and specific clauses that match the fields in the schema.

3. Extract the required entities from the document text. Make sure to capture all relevant information for each field specified in the schema.

4. Check item by item. This is an educational project, if you fail, many children will not learn correctly.

5. Double-check your work:
- Verify that all extracted information accurately reflects the content of the document
- Ensure no required fields are left empty unless the information is genuinely not present in the document

Remember, accuracy and adherence to the schema are crucial. Do not add any fields or information not specified in the schema, and do not omit any required fields unless the information is absent from the document text."""

        reduced_prompt = truncated_content + end_of_instructions

        # Recalculate the number of tokens
        new_num_tokens = len(encoding.encode(reduced_prompt))

        logger.info(f"Reduced prompt from {num_tokens} to {new_num_tokens} tokens")
        return reduced_prompt, new_num_tokens
    else:
        return prompt, num_tokens


def ner_extractor(content: str) -> str:
    """
    Named Entity Recognition (NER) extractor for contracts.

    Args:
        content (str): The content of the contract's body.

    Returns:
        str: A prompt for the model to extract entities, including the contract body.
    """
    logger.info("Generating NER extraction prompt")
    contract_extraction_prompt = """You are a document entity extraction specialist. Your task is to carefully read a document, which most likely is a contract, extract specific information according to a provided schema, and present it in a structured JSON format. Follow these steps precisely:

1. Read the following document text carefully:
<document>
{CONTRACT_TEXT}
</document>

2. Analyze the document text thoroughly. Pay attention to key details such as names, dates, amounts, all clauses, all aliases, and specific clauses that match the fields in the schema.

3. Extract the required entities from the document text. Make sure to capture all relevant information for each field specified in the schema.

4. Check item by item. This is an educational project, if you fail, many children will not learn correctly.

5. Double-check your work:
- Verify that all extracted information accurately reflects the content of the document
- Ensure no required fields are left empty unless the information is genuinely not present in the document

Remember, accuracy and adherence to the schema are crucial. Do not add any fields or information not specified in the schema, and do not omit any required fields unless the information is absent from the document text.
"""
    combined_prompt = contract_extraction_prompt.format(CONTRACT_TEXT=content)
    adjusted_prompt, num_tokens = adjust_prompt_length(combined_prompt)

    logger.debug(f"Generated prompt of length: {num_tokens} tokens")
    return adjusted_prompt


def exponential_backoff(attempt):
    delay = min(INITIAL_RETRY_DELAY * (2**attempt) + random.uniform(0, 1), 300)
    logger.info(
        f"Calculated backoff delay: {delay:.2f} seconds for attempt {attempt + 1}"
    )
    return delay


async def process_item(json_obj, i, outfile, prompt_file):
    logger.info(f"Processing item {i+1}")
    processed_obj = json_obj.copy()

    if "body" in json_obj and isinstance(json_obj["body"], str):
        logger.info(
            f"Found 'body' field in JSON object for item {i+1}. Proceeding with extraction."
        )
        for attempt in range(MAX_RETRIES):
            try:
                prompt = ner_extractor(json_obj["body"])
                logger.info(f"Generated NER extraction prompt for item {i+1}")
                async with aiofiles.open(PROMPT_FILE, "a") as f:
                    await f.write(json.dumps({"prompt": prompt}) + "\n")
                logger.info(f"Sending prompt to OpenAI API for item {i+1}")

                try:
                    completion = await asyncio.wait_for(
                        client.beta.chat.completions.parse(
                            model="gpt-4o-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert at structured data extraction. You will be given unstructured text from a contract and should convert it into the given structure.",
                                },
                                {"role": "user", "content": prompt},
                            ],
                            response_format=ContractExtraction,
                        ),
                        timeout=600,
                    )
                    if completion and completion.choices[0].message.parsed:
                        logger.debug(
                            f"Received non-empty response from model for item {i+1}"
                        )
                        try:
                            extracted_data = completion.choices[0].message.parsed
                            processed_obj["processed_bodygpt"] = extracted_data.dict()
                            logger.info(
                                f"Successfully processed document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
                            )
                            break  # Success, exit retry loop
                        except Exception as e:
                            logger.error(
                                f"Failed to process parsed response for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1}): {str(e)}"
                            )
                    else:
                        logger.warning(
                            f"Empty response from model for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
                        )
                except asyncio.TimeoutError:
                    logger.error(f"API call timed out for item {i+1}")
                    processed_obj["processed_bodygpt"] = "ERROR! TIMEOUT!"
                    break  # Exit retry loop on timeout
            except Exception as e:
                logger.error(
                    f"Attempt {attempt + 1} failed for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1}): {str(e)}"
                )
                if attempt < MAX_RETRIES - 1:
                    delay = exponential_backoff(attempt)
                    logger.info(f"Retrying in {delay:.2f} seconds for item {i+1}...")
                    await asyncio.sleep(delay)
                else:
                    logger.warning(
                        f"Max retries reached for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1}). Passing."
                    )
                    processed_obj["processed_bodygpt"] = "ERROR! MAX RETRIES REACHED!"
    else:
        logger.warning(
            f"No 'body' field found or 'body' is not a string for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})."
        )
        processed_obj["processed_bodygpt"] = "ERROR! NO BODY!"

    logger.info(f"Writing processed object to output file for item {i+1}")
    await outfile.write(json.dumps(processed_obj) + "\n")
    await outfile.flush()
    logger.debug(
        f"Wrote processed object to output file for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
    )

    return i + 1


async def process_file(input_file: str, output_file: str, num_items: int = None):
    logger.info(f"Starting to process file: {input_file}")
    logger.info(f"Output will be written to: {output_file}")

    async with aiofiles.open(
        input_file, "r", encoding="utf-8"
    ) as infile, aiofiles.open(
        output_file, "a", encoding="utf-8"
    ) as outfile, aiofiles.open(
        STARTING_FILE, "a+", encoding="utf-8"
    ) as starting_file, aiofiles.open(
        PROMPT_FILE, "a", encoding="utf-8"
    ) as prompt_file:
        # Get the last processed item
        await starting_file.seek(0)
        last_processed = await starting_file.readlines()
        last_item = (
            int(last_processed[-1].strip()) if last_processed else STARTING_POINT
        )

        # Skip to the last processed item
        for _ in range(last_item):
            await infile.readline()

        pbar = tqdm(total=num_items, initial=last_item, desc="Processing items")

        async def worker(queue):
            while True:
                item = await queue.get()
                if item is None:
                    break
                i, line = item
                try:
                    json_obj = json.loads(line)
                    last_processed = await process_item(
                        json_obj, i, outfile, prompt_file
                    )
                    await starting_file.write(f"{last_processed}\n")
                    await starting_file.flush()
                    pbar.update(1)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Error decoding JSON for item {i+1}. Skipping line."
                    )
                finally:
                    queue.task_done()

        queue = asyncio.Queue()
        workers = [asyncio.create_task(worker(queue)) for _ in range(NO_WORKERS)]

        i = last_item
        async for line in infile:
            if num_items is not None and i >= num_items:
                logger.info(
                    f"Reached specified limit of {num_items} items. Stopping processing."
                )
                break
            await queue.put((i, line))
            i += 1

        # Signal workers to exit
        for _ in range(NO_WORKERS):
            await queue.put(None)

        # Wait for all workers to complete
        await asyncio.gather(*workers)

    logger.info("Finished processing file")


async def main():
    try:
        await process_file(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        logger.info("Script execution completed.")


if __name__ == "__main__":
    asyncio.run(main())
