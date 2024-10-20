import json
import logging
import random
import signal
import time
from typing import List

import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
INPUT_FILE = "secba_nu_input.jsonl"  # secba.ndjson
OUTPUT_FILE = "secba_process_nu.jsonl"
PROMPT_FILE = "zzpromptsgpt.jsonl"
STARTING_FILE = "startingfilenu.jsonl"

prompt_length = 1000000
MAX_RETRIES = 50
INITIAL_RETRY_DELAY = 1  # seconds
STARTING_POINT = 0

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


def adjust_prompt_length(tokenizer, prompt, max_tokens=115000):
    # Count the number of tokens in the prompt
    num_tokens = len(tokenizer.encode(prompt))

    # If the number of tokens exceeds the maximum, reduce the prompt size
    if num_tokens > max_tokens:
        logger.warning(
            f"Prompt exceeds maximum length of {max_tokens} tokens. Truncating..."
        )

        # Find the start of the CONTRACT_TEXT
        contract_start = prompt.find("<document>") + len("<document>")
        contract_end = prompt.find("</document>")

        # Calculate how many tokens we need to remove
        tokens_to_remove = num_tokens - max_tokens

        # Encode only the CONTRACT_TEXT
        contract_text = prompt[contract_start:contract_end]
        contract_tokens = tokenizer.encode(contract_text)

        # Remove tokens from the end of CONTRACT_TEXT
        truncated_contract_tokens = contract_tokens[:-tokens_to_remove]
        truncated_contract = tokenizer.decode(truncated_contract_tokens)

        # Reconstruct the prompt
        truncated_prompt = (
            prompt[:contract_start] + truncated_contract + prompt[contract_end:]
        )

        # Recalculate the number of tokens
        new_num_tokens = len(tokenizer.encode(truncated_prompt))

        logger.info(f"Reduced prompt from {num_tokens} to {new_num_tokens} tokens")
        return truncated_prompt, new_num_tokens
    else:
        return prompt, num_tokens


def ner_extractor(tokenizer, content: str) -> str:
    """
    Named Entity Recognition (NER) extractor for contracts.

    Args:
        tokenizer: The tokenizer to use for token counting.
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
    adjusted_prompt, num_tokens = adjust_prompt_length(tokenizer, combined_prompt)

    logger.debug(f"Generated prompt of length: {num_tokens} tokens")
    return adjusted_prompt


def predict_NuExtract(
    model,
    tokenizer,
    texts,
    template,
    batch_size=1,
    # max_length=10000,
    max_new_tokens=4000,
):
    template = json.dumps(json.loads(original_template), indent=4)
    prompts = [
        f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>"""
        for text in texts
    ]

    outputs = []
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            batch_encodings = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=131072,
            ).to(model.device)

            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return [output.split("<|output|>")[1] for output in outputs]


def exponential_backoff(attempt):
    delay = min(INITIAL_RETRY_DELAY * (2**attempt) + random.uniform(0, 1), 300)
    logger.info(
        f"Calculated backoff delay: {delay:.2f} seconds for attempt {attempt + 1}"
    )
    return delay


def process_file(input_file: str, output_file: str, num_items: int = None):
    """
    Processes the input file (NDJSON or JSONL), extracts data using the ner_extractor,
    and writes the results to the output JSONL file.

    Args:
        input_file (str): Path to the input file (NDJSON or JSONL).
        output_file (str): Path to the output JSONL file.
        num_items (int, optional): Number of items to process. If None, process all items.
    """
    logger.info(f"Starting to process file: {input_file}")
    logger.info(f"Output will be written to: {output_file}")

    model_name = "numind/NuExtract-v1.5"
    device = "cuda"
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    template = json.dumps(ContractExtraction.schema()["properties"])
    print(template)
    with open(input_file, "r", encoding="utf-8") as infile, open(
        output_file, "a", encoding="utf-8"
    ) as outfile, open(STARTING_FILE, "a+", encoding="utf-8") as starting_file:
        # Get the last processed item
        starting_file.seek(0)
        last_processed = starting_file.readlines()
        last_item = (
            int(last_processed[-1].strip()) if last_processed else STARTING_POINT
        )

        # Skip to the last processed item
        for _ in range(last_item):
            next(infile, None)

        pbar = tqdm(total=num_items, initial=last_item, desc="Processing items")

        for i, line in enumerate(infile, start=last_item):
            if num_items is not None and i >= num_items:
                logger.info(
                    f"Reached specified limit of {num_items} items. Stopping processing."
                )
                break

            logger.info(f"Processing item {i+1}")
            try:
                json_obj = json.loads(line)
                logger.debug(f"Successfully loaded JSON object for item {i+1}")
            except json.JSONDecodeError:
                logger.warning(f"Error decoding JSON for item {i+1}. Skipping line.")
                continue

            processed_obj = json_obj.copy()
            logger.debug("Created copy of JSON object for processing")

            if "body" in json_obj and isinstance(json_obj["body"], str):
                logger.info(
                    f"Found 'body' field in JSON object for item {i+1}. Proceeding with extraction."
                )
                for attempt in range(MAX_RETRIES):
                    try:
                        prompt = ner_extractor(tokenizer, json_obj["body"])
                        logger.info(f"Generated NER extraction prompt for item {i+1}")
                        with open(PROMPT_FILE, "a") as f:
                            f.write(json.dumps({"prompt": prompt}) + "\n")
                        logger.info(f"Sending prompt to NuExtract model for item {i+1}")
                        signal.alarm(1000)  # Set a 10-minute timeout
                        try:
                            prediction = predict_NuExtract(
                                model, tokenizer, [prompt], template
                            )[0]
                            signal.alarm(0)  # Disable the alarm if successful
                            if prediction:
                                logger.debug(
                                    f"Received non-empty response from model for item {i+1}"
                                )
                                try:
                                    extracted_data = json.loads(prediction)
                                    processed_obj["processed_body_nu"] = extracted_data
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
                        except TimeoutException:
                            logger.error(f"API call timed out for item {i+1}")
                            processed_obj["processed_body_nu"] = "ERROR! TIMEOUT!"
                            break  # Exit retry loop on timeout
                    except Exception as e:
                        logger.error(
                            f"Attempt {attempt + 1} failed for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1}): {str(e)}"
                        )
                        if attempt < MAX_RETRIES - 1:
                            delay = exponential_backoff(attempt)
                            logger.info(
                                f"Retrying in {delay:.2f} seconds for item {i+1}..."
                            )
                            time.sleep(delay)
                        else:
                            logger.warning(
                                f"Max retries reached for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1}). Passing."
                            )
                            processed_obj["processed_body_nu"] = (
                                "ERROR! MAX RETRIES REACHED!"
                            )
            else:
                logger.warning(
                    f"No 'body' field found or 'body' is not a string for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})."
                )
                processed_obj["processed_body_nu"] = "ERROR! NO BODY!"

            logger.info(f"Writing processed object to output file for item {i+1}")
            json.dump(processed_obj, outfile)
            outfile.write("\n")
            outfile.flush()  # Ensure data is written immediately
            logger.debug(
                f"Wrote processed object to output file for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
            )

            # Update the starting file with the last processed item
            starting_file.write(f"{i+1}\n")
            starting_file.flush()

            pbar.update(1)

    logger.info("Finished processing file")


if __name__ == "__main__":
    try:
        process_file(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        logger.info("Script execution completed.")
