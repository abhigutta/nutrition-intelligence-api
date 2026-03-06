# Nutrition Intelligence Scanner API

AWS serverless API that analyzes food nutrition using OpenAI GPT-5.2-mini and stores results in DynamoDB.

## Architecture

```
Client → API Gateway (API Key auth) → Lambda (Python 3.12) → OpenAI GPT-5.2-mini
                                            ↓
                                       DynamoDB
                                    (foodItem + version)
```

## DynamoDB Schema

| Key | Type | Example |
|-----|------|---------|
| `foodItem` (Partition Key) | String | `eggs_spinach` |
| `version` (Sort Key) | String | `v1` |

## Prerequisites

- AWS CLI configured with credentials
- AWS SAM CLI installed
- OpenAI API key with access to `gpt-5.2-mini`

## Deploy

```bash
chmod +x deploy.sh
./deploy.sh
```

Or manually:

```bash
sam build --template-file template.yaml --use-container

sam deploy \
  --stack-name nutrition-intelligence-api \
  --capabilities CAPABILITY_IAM \
  --parameter-overrides "OpenAIApiKey=sk-your-key-here" \
  --resolve-s3
```

## Usage

```bash
# Basic analysis
curl -X POST https://<api-id>.execute-api.<region>.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: <your-api-key>" \
  -d '{"foodItems": "spinach, eggs"}'

# With version and user profile
curl -X POST https://<api-id>.execute-api.<region>.amazonaws.com/prod/analyze \
  -H "Content-Type: application/json" \
  -H "x-api-key: <your-api-key>" \
  -d '{
    "foodItems": "salmon, broccoli, rice",
    "version": "v2",
    "userProfile": {
      "age": 30,
      "goal": "muscle_gain",
      "allergies": ["shellfish"]
    }
  }'
```

## Request Body

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `foodItems` | Yes | — | Comma-separated food items |
| `version` | No | `v1` | Version tag for DynamoDB sort key |
| `userProfile` | No | `null` | User health profile object |
| `oldContext` | No | `null` | Previous analysis for follow-ups |

## Cleanup

```bash
aws cloudformation delete-stack --stack-name nutrition-intelligence-api
```
