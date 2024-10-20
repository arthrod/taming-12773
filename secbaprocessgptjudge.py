import argparse
import asyncio
import json
import logging
import random
import signal
from typing import Any, Dict, List

import aiofiles
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm import tqdm

# Configuration
INPUT_FILE = (
    "/home/arthrod/secbaprocess/nu2/secba_process_nu.jsonl"  # "secba_processgpt.jsonl"
)
OUTPUT_FILE = "secba_process_judge3.jsonl"
PROMPT_FILE = "zprompjudge.jsonl"
STARTING_FILE = "startjudge3.jsonl"

MAX_TOKENS = 113000

MAX_RETRIES = 50
INITIAL_RETRY_DELAY = 1  # seconds
STARTING_POINT = 0
NO_WORKERS = 40

# Ignore SIGHUP (hangup signal)
signal.signal(signal.SIGHUP, signal.SIG_IGN)

# Load environment variables from the specified path
load_dotenv("/home/arthrod/.env/.env")

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("gpt_judge_detailed.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

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


original_template = {
    "ContractExtraction": {
        "document_name": "",
        "document_type": "",
        "agreement_name": "",
        "parties": [],
        "agreement_date": "",
        "effective_date": "",
        "expiration_date": "",
        "contract_amount": "",
        "currency": "",
        "payment_terms": "",
        "governing_law": "",
        "jurisdiction": "",
        "signatories": [],
        "amendment_history": [],
        "contract_term_period": "",
        "renewal_terms": "",
        "summary": "",
        "operative_clause": "",
        "confidentiality": "",
        "termination": "",
        "indemnification": "",
        "limitation_of_liability": "",
        "intellectual_property": "",
        "dispute_resolution": "",
        "force_majeure": "",
        "assignment": "",
        "non_compete": "",
        "non_solicitation": "",
        "warranties": "",
        "insurance": "",
        "audit_rights": "",
        "data_protection": "",
        "compliance_with_laws": "",
        "affiliate_license_licensee": "",
        "anti_assignment": "",
        "change_of_control": "",
        "claims": "",
        "competitive_restriction": "",
        "covenant_not_to_sue": "",
        "early_termination": "",
        "engagement": "",
        "entire_agreement": "",
        "escrow": "",
        "exclusivity": "",
        "fees": "",
        "ip_ownership": "",
        "license_grant": "",
        "liquidated_damages": "",
        "minimum_commitment": "",
        "payment_and_fees": "",
        "price_restrictions": "",
        "renewal_term": "",
        "representations_and_warranties": "",
        "scope_of_use": "",
        "services": "",
        "severability_clause": "",
        "survival": "",
        "taxes": "",
        "term": "",
        "termination_for_convenience": "",
        "third_party_beneficiary": "",
        "waiver": "",
        "average_confidence": "",
        "total_sections": "",
    }
}


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("API call timed out")


# Set the signal handler and a 10-minute alarm
signal.signal(signal.SIGALRM, timeout_handler)


def truncate_text(text: str, max_tokens: int) -> str:
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    logger.warning(
        f"Prompt exceeds maximum length of {max_tokens} tokens. Truncating..."
    )

    # Find the start of the contract body
    contract_start = text.find("<whole_body_of_contract>") + len(
        "<whole_body_of_contract>"
    )
    contract_end = text.find("</whole_body_of_contract>")

    # Calculate how many tokens we need to remove
    tokens_to_remove = len(tokens) - max_tokens

    # Encode only the contract body
    contract_text = text[contract_start:contract_end]
    contract_tokens = encoding.encode(contract_text)

    # Remove tokens from the end of the contract body
    truncated_contract_tokens = contract_tokens[:-tokens_to_remove]
    truncated_contract = encoding.decode(truncated_contract_tokens)

    # Reconstruct the prompt
    truncated_text = text[:contract_start] + truncated_contract + text[contract_end:]

    # Recalculate the number of tokens
    new_num_tokens = len(encoding.encode(truncated_text))

    logger.info(f"Reduced prompt from {len(tokens)} to {new_num_tokens} tokens")
    return truncated_text


def create_gpt_prompt(body: str, processed_body_nu: Dict[str, Any]) -> str:
    prompt = f"""<instructions>
You are an expert judge evaluating the quality of information extraction from contracts. Your task is to assess the accuracy and completeness of the extracted information compared to the original contract text. Please provide:

1. An overall grade (0-100) based on the accuracy and completeness of the extraction.
2. The number of false positives (information that was extracted but wan't present in the original contract).
3. The number of missed items (information that should have been extracted but wasn't). BE CAREFUL! An item is considered missing only if it is not present in the contract AND is present in the schema. If the item is in the schema but not present in the contract, it is NOT considered a missed item. The schema is what we want to extract, but if it is not in the contract, then it is correct if not extracted. Consider null as the same as not present.
4. Wrong items (information that was extracted incorrectly).
5. A brief explanation of your assessment.

Provide your assessment in the following format:
1. Overall Grade: [Your grade];
2. Number of Wrong Items: [Your count];
2.1. Wrong Items: [Your list of items];
3. Number of Missed Items: [Your count];
3.1. Missed Items: [Your list of items];
4. Explanation: [Your explanation];

Example:

Item to be judged:
"
  "ContractExtraction": 
    "document_name": "Simple Consulting Agreement",
    "document_type": "Consulting Agreement",
    "agreement_name": "Consulting Agreement",
    "parties": ["ABC Corp", "Jane Doe"],
    "agreement_date": "May 15, 2023",
    "effective_date": "June 1, 2023",
    "expiration_date": "December 31, 2023",
    "contract_amount": 5000,
    "currency": "USD",
    "payment_terms": "Monthly",
    "governing_law": "New York State",
    "jurisdiction": null,
    "signatories": [],
    "amendment_history": [],
    "contract_term_period": "7 months",
    "renewal_terms": null,
    "summary": "This is a consulting agreement between ABC Corp and Jane Doe for marketing strategy services.",
    "operative_clause": "Consultant will provide marketing strategy services to Client.",
    "confidentiality": "Consultant agrees to keep all Client information confidential.",
    "termination": "",
    "indemnification": null,
    "limitation_of_liability": null,
    "intellectual_property": null,
    "dispute_resolution": null,
    "force_majeure": null,
    "assignment": null,
    "non_compete": null,
    "non_solicitation": null,
    "warranties": null,
    "insurance": null,
    "audit_rights": null,
    "data_protection": null,
    "compliance_with_laws": null,
    "affiliate_license_licensee": null,
    "anti_assignment": null,
    "change_of_control": null,
    "claims": null,
    "competitive_restriction": null,
    "covenant_not_to_sue": null,
    "early_termination": null,
    "engagement": "Consultant will provide marketing strategy services to Client.",
    "entire_agreement": null,
    "escrow": null,
    "exclusivity": null,
    "fees": null,
    "ip_ownership": null,
    "license_grant": null,
    "liquidated_damages": null,
    "minimum_commitment": null,
    "payment_and_fees": "Client will pay Consultant $2,000 per month.",
    "price_restrictions": null,
    "renewal_term": null,
    "representations_and_warranties": null,
    "scope_of_use": null,
    "services": "Marketing strategy services",
    "severability_clause": null,
    "survival": null,
    "taxes": null,
    "term": "This Agreement begins on June 1, 2023, and ends on December 31, 2023.",
    "termination_for_convenience": "Either party may terminate this Agreement with 30 days' notice.",
    "third_party_beneficiary": null,
    "waiver": null,
    "average_confidence": 0.9,
    "total_sections": 6"

Whole body of the contract:
"SIMPLE CONSULTING AGREEMENT

This Consulting Agreement (the "Agreement") is made on May 15, 2023, between ABC Corp ("Client") and Jane Doe ("Consultant").

1. Services: Consultant will provide marketing strategy services to Client.

2. Term: This Agreement begins on June 1, 2023, and ends on December 31, 2023.

3. Compensation: Client will pay Consultant $5,000 per month.

4. Confidentiality: Consultant agrees to keep all Client information confidential.

5. Termination: Either party may terminate this Agreement with 30 days' notice.

6. Governing Law: This Agreement is governed by the laws of New York State.

Signed:

_________________    _________________
ABC Corp             Jane Doe"

Assessment:
"1. Overall Grade: 75;
2. Number of Wrong Items: 1;
2.1. Wrong Items: [Sec. 3, Compensation];
3. Number of Missed Items: 1;
3.1. Missed Items: [Sec. 5 Termination, Signatories];
4. Wrong Items: 1;
4. Explanation: The extraction is generally accurate and captures most of the key information from the agreement. However, it missed a few items that are present in the contract and schema:
 - Termination is present in the agreement: Either party may terminate this Agreement with 30 days' notice.
 - Compensation should have been $5,000.
 - The signatories are not listed, though spaces for signatures are       
      provided.
   - The payment terms (monthly) are not explicitly extracted.
   Despite these minor omissions, the extraction correctly identified the document type, parties, agreement date, effective date, expiration date, contract amount, currency, governing law, and key clauses such as confidentiality and termination. The extraction demonstrates good accuracy for the information it did capture."

As you can see, the schema usually has more items than the actual agreement. If it is NOT in the agreement and it is in the schema: please ignore. A missing item is an item that was listed in the schema AND listed in the agreement, but not in the <item_to_be_judged>.

Follow these instructions carefully and ensure your response follows the specified format.
</instructions>

<schema_of_items_that_should_have_been_identified_if_present_in_the_agreement>
{json.dumps(ContractExtraction.schema(), indent=2)}
</schema_of_items_that_should_have_been_identified_if_present_in_the_agreement>

<object_of_judgment>
<item_to_be_judged>
{json.dumps(processed_body_nu, indent=2)}
</item_to_be_judged>
</object_of_judgment>

<whole_body_of_contract>
{body}
</whole_body_of_contract>
"""
    return truncate_text(prompt, MAX_TOKENS)


def exponential_backoff(attempt):
    delay = min(INITIAL_RETRY_DELAY * (2**attempt) + random.uniform(0, 1), 300)
    logger.info(
        f"Calculated backoff delay: {delay:.2f} seconds for attempt {attempt + 1}"
    )
    return delay


async def process_item(json_obj, i, outfile, client, dry_run=False):
    logger.info(f"Processing item {i+1}")
    processed_obj = json_obj.copy()

    if (
        "body" in json_obj
        and isinstance(json_obj["body"], str)
        and "processed_body_nu" in json_obj
    ):
        logger.info(
            f"Found 'body' and 'processed_body_nu' fields in JSON object for item {i+1}. Proceeding with extraction."
        )
        for attempt in range(MAX_RETRIES):
            try:
                prompt = create_gpt_prompt(
                    json_obj["body"], json_obj["processed_body_nu"]
                )
                logger.info(f"Generated nu prompt for item {i+1}")
                async with aiofiles.open(PROMPT_FILE, "a") as f:
                    await f.write(json.dumps({"prompt": prompt}) + "\n")
                logger.info(f"Sending prompt to nu API for item {i+1}")

                if dry_run:
                    logger.info(f"Dry run: Simulating nu API call for item {i+1}")
                    assessment = "DRY RUN: Simulated nu assessment"
                else:
                    try:
                        message = await asyncio.wait_for(
                            client.chat.completions.create(
                                model="gpt-4o-mini",
                                messages=[
                                    {
                                        "role": "user",
                                        "content": prompt,
                                    }
                                ],
                                temperature=0,
                            ),
                            timeout=6000,
                        )
                        if message and message.choices[0].message.content:
                            logger.debug(
                                f"Received non-empty response from nu for item {i+1}"
                            )
                            assessment = message.choices[0].message.content
                        else:
                            logger.warning(
                                f"Empty response from nu for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
                            )
                            assessment = "ERROR! EMPTY RESPONSE!"
                    except asyncio.TimeoutError:
                        logger.error(f"API call timed out for item {i+1}")
                        assessment = "ERROR! TIMEOUT!"

                processed_obj["gpt_assessment_nu"] = assessment
                print(
                    f"Successfully processed document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
                )
                print(f"GPT Assessment: {assessment}")
                break  # Success, exit retry loop
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
                    processed_obj["gpt_assessment_nu"] = "ERROR! MAX RETRIES REACHED!"
    else:
        logger.warning(
            f"No 'body' or 'processed_body_nu' field found or 'body' is not a string for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})."
        )
        processed_obj["gpt_assessment_nu"] = "ERROR! MISSING REQUIRED FIELDS!"

    logger.info(f"Writing processed object to output file for item {i+1}")
    await outfile.write(json.dumps(processed_obj) + "\n")
    await outfile.flush()
    logger.debug(
        f"Wrote processed object to output file for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
    )

    return i + 1


async def process_file(
    input_file: str,
    output_file: str,
    client: AsyncOpenAI,
    num_items: int = None,
    dry_run: bool = False,
):
    logger.info(f"Starting to process file: {input_file}")
    logger.info(f"Output will be written to: {output_file}")
    if dry_run:
        logger.info("Running in DRY RUN mode. No actual API calls will be made.")

    async with aiofiles.open(
        input_file, "r", encoding="utf-8"
    ) as infile, aiofiles.open(
        output_file, "a", encoding="utf-8"
    ) as outfile, aiofiles.open(STARTING_FILE, "a+", encoding="utf-8") as starting_file:
        # Get the last processed item
        await starting_file.seek(0)
        last_processed = await starting_file.read()
        last_item = (
            int(last_processed.strip().split("\n")[-1])
            if last_processed
            else STARTING_POINT
        )

        # Skip to the last processed item
        for _ in range(last_item):
            await infile.readline()

        pbar = tqdm(total=num_items, initial=last_item, desc="Processing items")

        queue = asyncio.Queue()

        async def worker():
            while True:
                item = await queue.get()
                if item is None:
                    break
                i, line = item
                try:
                    json_obj = json.loads(line)
                    await process_item(json_obj, i, outfile, client, dry_run)
                    await starting_file.write(f"{i+1}\n")
                    await starting_file.flush()
                    pbar.update(1)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Error decoding JSON for item {i+1}. Skipping line."
                    )
                finally:
                    queue.task_done()

        workers = [asyncio.create_task(worker()) for _ in range(NO_WORKERS)]

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
    parser = argparse.ArgumentParser(description="Process contracts with GPT judge.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without making actual API calls",
    )
    parser.add_argument(
        "--test", type=int, help="Number of items to process for a test run"
    )
    args = parser.parse_args()

    try:
        async with AsyncOpenAI() as client:
            if args.test:
                print(f"Running in test mode. Processing {args.test} items.")
                await process_file(
                    INPUT_FILE,
                    OUTPUT_FILE,
                    client,
                    num_items=args.test,
                    dry_run=args.dry_run,
                )
            else:
                await process_file(
                    INPUT_FILE, OUTPUT_FILE, client, dry_run=args.dry_run
                )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        logger.info("Script execution completed.")


if __name__ == "__main__":
    asyncio.run(main())
