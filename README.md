# Taming 12,775 Contracts for $39.26!

## Overview
This repository contains the scripts and methodology used to analyze 12,772 contracts extracted from SEC filings using AI models. The project demonstrates the potential of Large Language Models (LLMs) in extracting metadata from legal documents at scale and at a fraction of the cost of traditional methods.

You can check my blog at Medium at synthetic.lawyer 

## Key Features
- Processes 12,773 agreements from SEC EDGAR database
- Utilizes NuExtract, GPT and Google's Gemini for metadata extraction
- Implements structured outputs for consistent data parsing
- Achieves a cost of approximately $0.003 per agreement

## Dataset
The dataset used in this project consists of 12,773 agreements extracted from SEC filings, each labeled as "Exhibit 10" in the EDGAR database. These documents represent a variety of material contracts that public companies are required to disclose.

The full dataset is available on Hugging Face: arthrod/taming-12773 (https://huggingface.co/datasets/arthrod/tqming-12773)

## Methodology
1. **Data Collection**: Agreements were sourced from the SEC's EDGAR database.
2. **AI Models**: The project primarily uses GPT and Google's Gemini for metadata extraction.
3. **Structured Outputs**: A schema was defined to extract key metadata elements from each agreement.
4. **Processing**: Each document was processed through the AI models to extract structured data.

## Scripts
This repository contains the following key scripts:

1. Test script.
2. NuExtract-1.5 (hosted on Shadeform using vLLM), GPT-4o-mini (OpenAI Async) and Flash 1.5 (VertexAPI).
3. Judge script (GPT-4o-mini).
4. Normalization script.

## Results
- Successfully processed 12,773 documents
- Total cost: $39.26 ($0.003 per agreement)
- Extracted metadata includes:
  - Agreement type
  - Parties involved
  - Effective date
  - Termination date
  - Governing law
  - And more (see full schema in any of the scripts)

## Future Work
- Integration with Claude and other AI models
- Expansion of the metadata schema
- Performance comparison between different AI models
- Exploration of use cases in due diligence and contract management

## Contributing
Contributions to this project are welcome! Please feel free to submit a Pull Request.

## License
MIT

## Contact
For questions or feedback, please email me at arthur@umich.edu

---

Remember to star this repo if you find it useful, and happy coding!
