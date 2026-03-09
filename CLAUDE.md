# Nutrition Intelligence Scanner API

## Overview
AWS serverless API that analyzes food nutrition using OpenAI GPT-5.2-mini and stores results in DynamoDB. Deployed via AWS SAM.

## Architecture
- **Lambda** (Python 3.13): `src/app.py` — single handler `lambda_handler`
- **DynamoDB**: Two tables
  - `NutritionAnalysis` — stores analysis results (PK: `foodItem`, SK: `version`)
  - `NutritionRateLimit` — per-device daily rate limiting (PK: `deviceId`, SK: `date`, TTL-enabled)
- **API Gateway**: REST API with API key auth, usage plan (1000/month, 5 rps, burst 10)
- **Secrets Manager**: Stores OpenAI API key

## Key Files
- `template.yaml` — SAM/CloudFormation template (all infra)
- `src/app.py` — Lambda function code
- `src/requirements.txt` — Python dependencies (openai, boto3)
- `deploy.sh` — Build & deploy script

## Build & Deploy
```bash
sam build --template-file template.yaml --use-container
sam deploy --stack-name nutrition-intelligence-api --capabilities CAPABILITY_IAM --parameter-overrides "OpenAIApiKey=<key>" --resolve-s3
```
Or use `./deploy.sh` which handles bucket creation and prompts for the key.

## Development Notes
- OpenAI client is cached globally for Lambda warm starts
- Rate limit: 50 requests/device/day (configurable via `RATE_LIMIT_MAX` env var)
- Food names are normalized and sorted for consistent DynamoDB keys (e.g., "Spinach, EGGS" -> "eggs_spinach")
- `.aws-sam/` directory contains build artifacts — do not edit directly
