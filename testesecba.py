import json
import logging

# Configuration
INPUT_FILE = "secba_process.jsonl"

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_script.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def read_and_print_bodies(input_file: str, num_items: int = 2):
    """
    Reads the specified number of items from the input file and prints their 'body' content.

    Args:
        input_file (str): Path to the input JSONL file.
        num_items (int): Number of items to process (default is 2).
    """
    logger.info(f"Starting to read file: {input_file}")

    bodies_found = 0
    with open(input_file, "r", encoding="utf-8") as infile:
        for i, line in enumerate(infile):
            if bodies_found >= num_items:
                break

            logger.info(f"Processing item {i+1}")
            try:
                json_obj = json.loads(line)
                logger.debug(f"Successfully loaded JSON object for item {i+1}")

                if "body" in json_obj and isinstance(json_obj["body"], str):
                    body_content = json_obj["body"].strip()
                    if body_content:
                        logger.info(f"Found non-empty 'body' field in item {i+1}")
                        print(f"\n--- Body content for item {i+1} ---")
                        print(f"{body_content[:500]}...")  # Print first 500 characters
                        print("--- End of body content ---\n")
                        bodies_found += 1
                    else:
                        logger.warning(f"'body' field is empty for item {i+1}")
                else:
                    logger.warning(
                        f"No 'body' field found or 'body' is not a string for item {i+1}"
                    )
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON for item {i+1}. Skipping line.")
                continue

    logger.info(f"Finished processing. Found {bodies_found} non-empty body fields.")


if __name__ == "__main__":
    try:
        read_and_print_bodies(INPUT_FILE)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
    finally:
        logger.info("Script execution completed.")
