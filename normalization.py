import jsonlines

# Input and output file paths
input_file = "/home/arthrod/secbaprocess/secba_processgpt.jsonl"
output_file = "/home/arthrod/secbaprocess/secba_processgpt_normal.jsonl"

# List of required keys
required_keys = [
    "document_name",
    "document_type",
    "agreement_name",
    "parties",
    "agreement_date",
    "effective_date",
    "expiration_date",
    "contract_amount",
    "currency",
    "payment_terms",
    "governing_law",
    "jurisdiction",
    "signatories",
    "amendment_history",
    "contract_term_period",
    "renewal_terms",
    "summary",
    "operative_clause",
    "confidentiality",
    "termination",
    "indemnification",
    "limitation_of_liability",
    "intellectual_property",
    "dispute_resolution",
    "force_majeure",
    "assignment",
    "non_compete",
    "non_solicitation",
    "warranties",
    "insurance",
    "audit_rights",
    "data_protection",
    "compliance_with_laws",
    "affiliate_license_licensee",
    "anti_assignment",
    "change_of_control",
    "claims",
    "competitive_restriction",
    "covenant_not_to_sue",
    "early_termination",
    "engagement",
    "entire_agreement",
    "escrow",
    "exclusivity",
    "fees",
    "ip_ownership",
    "license_grant",
    "liquidated_damages",
    "minimum_commitment",
    "payment_and_fees",
    "price_restrictions",
    "renewal_term",
    "representations_and_warranties",
    "scope_of_use",
    "services",
    "severability_clause",
    "survival",
    "taxes",
    "term",
    "termination_for_convenience",
    "third_party_beneficiary",
    "waiver",
    "average_confidence",
    "total_sections",
]


def find_key_in_dict(data, target_key_lower):
    """Recursively search for a key in nested dictionaries and lists."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() == target_key_lower:
                return value
            else:
                result = find_key_in_dict(value, target_key_lower)
                if result is not None:
                    return result
    elif isinstance(data, list):
        for item in data:
            result = find_key_in_dict(item, target_key_lower)
            if result is not None:
                return result
    return None


def process_jsonl(input_file, output_file):
    with jsonlines.open(input_file) as reader, jsonlines.open(
        output_file, mode="w"
    ) as writer:
        for obj in reader:
            if "processed_body" in obj and isinstance(obj["processed_body"], dict):
                processed_body = obj["processed_body"]
                existing_keys_lower = {k.lower() for k in processed_body.keys()}

                for key in required_keys:
                    key_lower = key.lower()
                    if key_lower not in existing_keys_lower:
                        value = find_key_in_dict(processed_body, key_lower)
                        processed_body[key] = value
                # No need to reassign obj['processed_body']
            writer.write(obj)


if __name__ == "__main__":
    process_jsonl(input_file, output_file)
    print(f"Processing complete. Output written to {output_file}")
