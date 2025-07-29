# GridAI ⚡

Your smart assistant for electrical grid planning and construction consulting.

## Features

- **Multi-format Document Support**: PDF, DOCX, XLSX, CSV, TXT, JSON
- **Domain-Specific AI**: Specialized for electrical engineering and construction
- **Chat Memory**: Maintains conversation context
- **Azure-Ready**: Configured for Microsoft Azure deployment
- **Rate Limiting**: Built-in OpenAI API rate limiting and retry logic

## Quick Start

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**

   ```bash
   # Create .env file
   OPENAI_API_KEY=your_openai_api_key_here
   ```

3. **Run Locally**
   ```bash
   streamlit run frontend.py
   ```

## Supported File Types

- **PDF**: Technical drawings, specifications, reports
- **DOCX**: Project documents, requirements
- **XLSX/XLS**: Data sheets, calculations, schedules
- **CSV**: Equipment lists, measurement data
- **TXT**: Plain text documents
- **JSON**: Configuration files, structured data

## Future: Azure Deployment

Azure deployment capabilities are built into the configuration but deployment scripts removed for now. The app is Azure-ready when needed.

## Engineering Domain Knowledge

Eric specializes in:

- Substation design and engineering
- Power system analysis and protection
- Construction project management
- Regulatory compliance (IEEE, NESC, NERC)
- Grid modernization technologies
- Safety protocols and risk assessment

### Key Standards Referenced

- **IEEE Std 80**: Substation grounding design
- **IEEE Std 142**: Industrial/commercial grounding
- **IEEE C37 series**: Protection and control
- **NESC**: Safety standards for electric supply

## Development Roadmap

- [x] Multi-format file support
- [x] Enhanced domain prompts
- [x] Azure deployment ready
- [x] Rate limiting and error handling
- [ ] Microsoft Azure AD authentication
- [ ] Role-based access control
- [ ] Advanced security features
- [ ] Production monitoring

## Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
AZURE_STORAGE_CONNECTION_STRING=your_storage_connection  # Optional
AZURE_STORAGE_CONTAINER=gridai-documents  # Optional
MAX_REQUESTS_PER_MINUTE=60  # Optional
MAX_TOKENS_PER_REQUEST=4000  # Optional
```

### Azure-Specific Settings

The app automatically detects Azure environment and adjusts configuration accordingly.

## Security Notes

- API keys are stored as environment variables
- File uploads are processed in temporary directories
- No persistent storage of uploaded documents (unless Azure Storage is configured)
- Rate limiting prevents API abuse

## Support

For technical questions about electrical engineering topics, ask Eric directly in the chat interface. For application support, refer to the deployment logs and Azure monitoring tools.
