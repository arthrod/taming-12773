import copy
import json
import logging
import os
import random
import signal
import time
from typing import Any, Dict

import tqdm
from dotenv import load_dotenv
from google.cloud import aiplatform
from tqdm import tqdm
from vertexai.generative_models import (
    GenerationConfig,
    GenerativeModel,
    HarmBlockThreshold,
    HarmCategory,
)

# Configuration
INPUT_FILE = "secba.ndjson"
OUTPUT_FILE = "secba_process.jsonl"
PROMPT_FILE = "zprompts.jsonl"  # Define the JSONL file for storing prompts
STARTING_FILE = "startingfile.jsonl"

prompt_length = 1000000
MAX_RETRIES = 50
INITIAL_RETRY_DELAY = 1  # seconds
STARTING_POINT = 60000

# Ignore SIGHUP (hangup signal)
signal.signal(signal.SIGHUP, signal.SIG_IGN)

load_dotenv()
# auth: run 'gcloud auth application-default'
os.environ["vertex_project"] = "secba-415004"
os.environ["vertex_location"] = "us-central1"

vertex_project = os.environ["vertex_project"]
vertex_location = os.environ["vertex_location"]

credentials = "/home/ubuntu/.config/gcloud/application_default_credentials.json"
## GET CREDENTIALS
file_path = "/home/ubuntu/.config/gcloud/application_default_credentials.json"

# Load the JSON file
with open(file_path, "r") as file:
    vertex_credentials = json.load(file)

# Convert to JSON string
vertex_credentials_json = json.dumps(vertex_credentials)
GOOGLE_APPLICATION_CREDENTIALS = json.dumps(vertex_credentials)

aiplatform.init(project=vertex_project, location=vertex_location)


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("API call timed out")


# Set the signal handler and a 5-minute alarm
signal.signal(signal.SIGALRM, timeout_handler)

# Load environment variables from .env file
load_dotenv()

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

model = GenerativeModel("gemini-1.5-flash-001")


gapic_schema_dict = {
    "type_": "OBJECT",
    "properties": {
        "document_metadata": {
            "type_": "OBJECT",
            "properties": {
                "Document_Name": {
                    "type_": "STRING",
                    "description": "The name of the document. This field cannot be null.",
                },
                "Document_Type": {"type_": "STRING"},
                "Agreement_Name": {
                    "type_": "STRING",
                    "description": "The name of the agreement. This field may be null.",
                    "nullable": True,
                },
                "Parties": {"type_": "ARRAY", "items": {"type_": "STRING"}},
                "Agreement_Date": {"type_": "STRING"},
                "Effective_Date": {"type_": "STRING"},
                "Expiration_Date": {"type_": "STRING", "nullable": True},
                "Contract_Amount": {"type_": "NUMBER"},
                "Currency": {"type_": "STRING"},
                "Payment_Terms": {"type_": "STRING", "nullable": True},
                "Governing_Law": {"type_": "STRING"},
                "Jurisdiction": {"type_": "STRING", "nullable": True},
                "Signatories": {"type_": "ARRAY", "items": {"type_": "STRING"}},
                "Amendment_History": {
                    "type_": "ARRAY",
                    "items": {"type_": "STRING"},
                    "nullable": True,
                },
                "Contract_Term_Period": {"type_": "STRING", "nullable": True},
                "Renewal_Terms": {"type_": "STRING", "nullable": True},
            },
            "required": ["Document_Name", "Parties"],
        },
        "summary": {
            "type_": "STRING",
            "description": "A 4-line summary of the document.",
        },
        "operative_clause": {
            "type_": "STRING",
            "description": "The full text of the operative clause.",
        },
        "contract_sections": {
            "type_": "OBJECT",
            "description": "Indicates whether specific sections are present in the contract. Provide only the section number and header, if available.",
            "properties": {
                "Confidentiality": {"type_": "STRING", "nullable": True},
                "Termination": {"type_": "STRING", "nullable": True},
                "Indemnification": {"type_": "STRING", "nullable": True},
                "Limitation_of_Liability": {"type_": "STRING", "nullable": True},
                "Intellectual_Property": {"type_": "STRING", "nullable": True},
                "Dispute_Resolution": {"type_": "STRING", "nullable": True},
                "Force_Majeure": {"type_": "STRING", "nullable": True},
                "Assignment": {"type_": "STRING", "nullable": True},
                "Non_Compete": {"type_": "STRING", "nullable": True},
                "Non_Solicitation": {"type_": "STRING", "nullable": True},
                "Warranties": {"type_": "STRING", "nullable": True},
                "Insurance": {"type_": "STRING", "nullable": True},
                "Audit_Rights": {"type_": "STRING", "nullable": True},
                "Data_Protection": {"type_": "STRING", "nullable": True},
                "Compliance_with_Laws": {"type_": "STRING", "nullable": True},
                "Affiliate_License_Licensee": {"type_": "STRING", "nullable": True},
                "Anti_Assignment": {"type_": "STRING", "nullable": True},
                "Change_of_Control": {"type_": "STRING", "nullable": True},
                "Claims": {"type_": "STRING", "nullable": True},
                "Competitive_Restriction": {"type_": "STRING", "nullable": True},
                "Covenant_Not_to_Sue": {"type_": "STRING", "nullable": True},
                "Early_Termination": {"type_": "STRING", "nullable": True},
                "Engagement": {"type_": "STRING", "nullable": True},
                "Entire_Agreement": {"type_": "STRING", "nullable": True},
                "Escrow": {"type_": "STRING", "nullable": True},
                "Exclusivity": {"type_": "STRING", "nullable": True},
                "Fees": {"type_": "STRING", "nullable": True},
                "IP_Ownership": {"type_": "STRING", "nullable": True},
                "License_Grant": {"type_": "STRING", "nullable": True},
                "Liquidated_Damages": {"type_": "STRING", "nullable": True},
                "Minimum_Commitment": {"type_": "STRING", "nullable": True},
                "Payment_and_Fees": {"type_": "STRING", "nullable": True},
                "Price_Restrictions": {"type_": "STRING", "nullable": True},
                "Renewal_Term": {"type_": "STRING", "nullable": True},
                "Representations_and_Warranties": {"type_": "STRING", "nullable": True},
                "Scope_of_Use": {"type_": "STRING", "nullable": True},
                "Services": {"type_": "STRING", "nullable": True},
                "Severability_Clause": {"type_": "STRING", "nullable": True},
                "Survival": {"type_": "STRING", "nullable": True},
                "Taxes": {"type_": "STRING", "nullable": True},
                "Term": {"type_": "STRING", "nullable": True},
                "Termination_for_Convenience": {"type_": "STRING", "nullable": True},
                "Third_Party_Beneficiary": {"type_": "STRING", "nullable": True},
                "Waiver": {"type_": "STRING", "nullable": True},
            },
        },
        "statistics": {
            "type_": "OBJECT",
            "properties": {
                "average_confidence": {"type_": "NUMBER"},
                "total_sections": {"type_": "INTEGER"},
            },
        },
    },
}


def helper_function_create_compatible_schema(
    schema_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """Converts a JsonSchema to a dict that both _convert_schema_dict_to_gapic and aiplatform_types.Schema accept."""
    gapic_schema_dict = copy.deepcopy(schema_dict)

    # Handle type conversion
    if "type" in gapic_schema_dict:
        if isinstance(gapic_schema_dict["type"], list):
            gapic_schema_dict["type_"] = [
                t.upper() for t in gapic_schema_dict.pop("type")
            ]
        else:
            gapic_schema_dict["type_"] = gapic_schema_dict.pop("type").upper()

    # Handle format conversion
    if "format" in gapic_schema_dict:
        gapic_schema_dict["format_"] = gapic_schema_dict.pop("format")

    # Handle nested structures
    for key in ["items", "additionalProperties", "not"]:
        if key in gapic_schema_dict:
            gapic_schema_dict[key] = helper_function_create_compatible_schema(
                gapic_schema_dict[key]
            )

    # Handle properties
    if "properties" in gapic_schema_dict:
        gapic_schema_dict["properties"] = {
            key: helper_function_create_compatible_schema(value)
            for key, value in gapic_schema_dict["properties"].items()
        }

    # Handle patternProperties
    if "patternProperties" in gapic_schema_dict:
        gapic_schema_dict["patternProperties"] = {
            key: helper_function_create_compatible_schema(value)
            for key, value in gapic_schema_dict["patternProperties"].items()
        }

    # Handle array-like structures
    for key in ["anyOf", "allOf", "oneOf", "prefixItems"]:
        if key in gapic_schema_dict:
            gapic_schema_dict[key] = [
                helper_function_create_compatible_schema(item)
                for item in gapic_schema_dict[key]
            ]

    # Handle definitions and $defs
    for key in ["definitions", "$defs"]:
        if key in gapic_schema_dict:
            gapic_schema_dict[key] = {
                subkey: helper_function_create_compatible_schema(subvalue)
                for subkey, subvalue in gapic_schema_dict[key].items()
            }

    # Handle if-then-else
    for key in ["if", "then", "else"]:
        if key in gapic_schema_dict:
            gapic_schema_dict[key] = helper_function_create_compatible_schema(
                gapic_schema_dict[key]
            )

    # List of all other possible JSON Schema keywords to preserve
    preserve_keys = [
        "title",
        "description",
        "default",
        "examples",
        "multipleOf",
        "maximum",
        "exclusiveMaximum",
        "minimum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxContains",
        "minContains",
        "maxProperties",
        "minProperties",
        "required",
        "dependentRequired",
        "const",
        "enum",
        "readOnly",
        "writeOnly",
        "deprecated",
        "contentEncoding",
        "contentMediaType",
        "contentSchema",
        "$schema",
        "$id",
        "$anchor",
        "$ref",
        "$recursiveRef",
        "$recursiveAnchor",
        "$vocabulary",
        "$comment",
        "dependentSchemas",
        "unevaluatedItems",
        "unevaluatedProperties",
        "propertyNames",
        "contains",
        "additionalItems",
    ]

    for key in preserve_keys:
        if key in gapic_schema_dict:
            gapic_schema_dict[key] = gapic_schema_dict[key]

    return gapic_schema_dict


response_schema = helper_function_create_compatible_schema(gapic_schema_dict)


def ner_extractor(content: str) -> str:
    """
    Named Entity Recognition (NER) extractor for contracts.

    Args:
        content (str): The content of the contract's body.

    Returns:
        str: A prompt for the model to extract entities, including the contract body.
    """
    logger.info("Generating NER extraction prompt")
    contract_extraction_prompt = """You are a document entity extraction specialist. Your task is to carefully read a doccument,which most likely is a contract, extract specific information according to a provided schema, and present it in a structured JSON format. Follow these steps precisely:

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
    end_of_instructions = """</document>

2. Analyze the document text thoroughly. Pay attention to key details such as names, dates, amounts, all clauses, all aliases, and specific clauses that match the fields in the schema.

3. Extract the required entities from the document text. Make sure to capture all relevant information for each field specified in the schema.

4. Check item by item. This is an educational project, if you fail, many children will not learn correctly.

5. Double-check your work:
- Verify that all extracted information accurately reflects the content of the document
- Ensure no required fields are left empty unless the information is genuinely not present in the document

Remember, accuracy and adherence to the schema are crucial. Do not add any fields or information not specified in the schema, and do not omit any required fields unless the information is absent from the document text.
"""
    if len(combined_prompt) > prompt_length:
        logger.warning(
            f"Prompt exceeds maximum length of {prompt_length}. Truncating..."
        )
        truncated_content = combined_prompt[: prompt_length - len(end_of_instructions)]
        combined_prompt = truncated_content + end_of_instructions

    logger.debug(f"Generated prompt of length: {len(combined_prompt)}")
    return combined_prompt


def exponential_backoff(attempt):
    delay = min(INITIAL_RETRY_DELAY * (2**attempt) + random.uniform(0, 1), 300)
    logger.info(
        f"Calculated backoff delay: {delay:.2f} seconds for attempt {attempt + 1}"
    )
    return delay


# Add these new variables


def process_ndjson(input_file: str, output_file: str, num_items: int = None):
    """
    Processes the input NDJSON file, extracts data using the ner_extractor,
    and writes the results to the output JSONL file.

    Args:
        input_file (str): Path to the input NDJSON file.
        output_file (str): Path to the output JSONL file.
        num_items (int, optional): Number of items to process. If None, process all items.
    """
    logger.info(f"Starting to process NDJSON file: {input_file}")
    logger.info(f"Output will be written to: {output_file}")

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
                        prompt = ner_extractor(json_obj["body"])
                        logger.info(f"Generated NER extraction prompt for item {i+1}")
                        with open(PROMPT_FILE, "a") as f:
                            f.write(json.dumps({"prompt": prompt}) + "\n")
                        logger.info(f"Sending prompt to GenerativeModel for item {i+1}")
                        signal.alarm(600)  # Set a 5-minute timeout
                        try:
                            response = model.generate_content(
                                prompt,
                                generation_config=GenerationConfig(
                                    response_mime_type="application/json",
                                    response_schema=response_schema,
                                ),
                                safety_settings={
                                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                                },
                            )
                            signal.alarm(0)  # Disable the alarm if successful
                            if response and response.text:
                                logger.debug(
                                    f"Received non-empty response from model for item {i+1}"
                                )
                                try:
                                    extracted_data = json.loads(response.text.strip())
                                    processed_obj["processed_body"] = extracted_data
                                    logger.info(
                                        f"Successfully processed document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
                                    )
                                    break  # Success, exit retry loop
                                except json.JSONDecodeError:
                                    logger.error(
                                        f"Failed to parse JSON response for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
                                    )
                            else:
                                logger.warning(
                                    f"Empty response from model for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})"
                                )
                        except TimeoutException:
                            logger.error(f"API call timed out for item {i+1}")
                            processed_obj["processed_body"] = "ERROR! TIMEOUT!"
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
                            processed_obj["processed_body"] = (
                                "ERROR! MAX RETRIES REACHED!"
                            )
            else:
                logger.warning(
                    f"No 'body' field found or 'body' is not a string for document {json_obj.get('accessionNo', 'Unknown')} (item {i+1})."
                )
                processed_obj["processed_body"] = "ERROR! NO BODY!"

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

    logger.info("Finished processing NDJSON file")


if __name__ == "__main__":
    try:
        process_ndjson(INPUT_FILE, OUTPUT_FILE)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        logger.info("Script execution completed.")
