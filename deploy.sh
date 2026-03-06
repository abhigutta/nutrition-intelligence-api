#!/bin/bash
# ──────────────────────────────────────────────────────────
# deploy.sh — Build & deploy the Nutrition Intelligence API
# ──────────────────────────────────────────────────────────
set -euo pipefail

STACK_NAME="nutrition-intelligence-api"
REGION="${AWS_REGION:-us-east-1}"
S3_BUCKET=""  # ← fill in your SAM deployment bucket

# ── Preflight checks ────────────────────────────────────
command -v sam >/dev/null 2>&1 || { echo "❌ AWS SAM CLI not installed. Install: https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html"; exit 1; }
command -v aws >/dev/null 2>&1 || { echo "❌ AWS CLI not installed."; exit 1; }

if [ -z "$S3_BUCKET" ]; then
    echo "⚠️  No S3_BUCKET set. Creating one..."
    S3_BUCKET="sam-deploy-${STACK_NAME}-$(date +%s)"
    aws s3 mb "s3://${S3_BUCKET}" --region "$REGION"
    echo "✅ Created bucket: $S3_BUCKET"
fi

# ── Prompt for OpenAI API Key ────────────────────────────
read -rsp "Enter your OpenAI API Key: " OPENAI_KEY
echo ""

# ── Build ────────────────────────────────────────────────
echo "🔨 Building..."
sam build --template-file template.yaml --use-container

# ── Deploy ───────────────────────────────────────────────
echo "🚀 Deploying stack: $STACK_NAME to $REGION..."
sam deploy \
    --stack-name "$STACK_NAME" \
    --s3-bucket "$S3_BUCKET" \
    --region "$REGION" \
    --capabilities CAPABILITY_IAM \
    --parameter-overrides "OpenAIApiKey=$OPENAI_KEY" \
    --no-confirm-changeset

# ── Print outputs ────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════"
echo "✅ Deployment complete!"
echo "════════════════════════════════════════════════════"
aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs" \
    --output table

# ── Retrieve API Key value ───────────────────────────────
API_KEY_ID=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --region "$REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiKeyId'].OutputValue" \
    --output text)

echo ""
echo "🔑 Your API Key value:"
aws apigateway get-api-key --api-key "$API_KEY_ID" --include-value --query "value" --output text --region "$REGION"

echo ""
echo "📌 Test with:"
echo "curl -X POST <API_ENDPOINT> \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -H 'x-api-key: <YOUR_API_KEY>' \\"
echo '  -d '\''{"foodItems": "spinach, eggs"}'\'''
