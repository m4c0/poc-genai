#!/bin/bash

curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer `cat ~/.openai`" \
  -d '{
  "model": "gpt-4o-mini",
  "store": true,
  "messages": [
    { "role": "developer",
      "content": "You are an robotic assistant" },
    { "role": "user",
      "content": "Example of calling OpenAI with python" }
  ]
}'

