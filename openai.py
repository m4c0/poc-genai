import http.client
import json
import os

class Exc(Exception):
  pass

with open(os.path.expanduser("~/.openai"), 'r') as file:
  api_key = file.read().strip()

conn = http.client.HTTPSConnection("api.openai.com")

headers = {
  'Authorization': f'Bearer {api_key}',
  'Content-Type': 'application/json'
}

data = json.dumps({
  "model": "gpt-4o-mini",
  "messages": [{
    "role": "developer",
    "content": "You are an automated assistant robot without personality"
  }, {
    "role": "user",
    "content": "List files in this repo"
  }],
  "tools": [{
    "type": "function",
    "function": {
      "name": "list_files",
      "description": "list files in current directory",
      "strict": True,
      "parameters": {
        "type": "object",
        "properties": {
          "path": {
            "type": "string",
            "description": "path to search"
          }
        },
        "required": ["path"],
        "additionalProperties": False
      }
    },
  }]
})

conn.request("POST", "/v1/chat/completions", data, headers)
resp = conn.getresponse()
resp_data = resp.read()
conn.close()

if resp.status != 200:
  raise Exc(f"Error: {resp.status}, {resp_data.decode()}")

msg = json.loads(resp_data)['choices'][0]['message']

class Funcs:
  def list_files(path):
    print(f"listing {path}")
    return os.listdir(".")

if msg['content']:
  print(msg['content'])

for tc in msg['tool_calls']:
  fn = tc['function']
  val = getattr(Funcs, fn['name'])(*json.loads(fn['arguments']))
  print(val)

