import http.client
import json
import os
import subprocess

class Exc(Exception):
  pass

with open(os.path.expanduser("~/.openai"), 'r') as file:
  api_key = file.read().strip()

conn = http.client.HTTPSConnection("api.openai.com")

headers = {
  'Authorization': f'Bearer {api_key}',
  'Content-Type': 'application/json'
}

messages = [{
  "role": "developer",
  "content": "You are an automated assistant robot without personality. You are dealing with a very capable human, so you don't need to explain to go into details of your reasoning."
}, {
  "role": "developer",
  "content": """
Output files may reside in a folder named 'out' or 'build'. Do not expose the directory name.

Usually, a C++ module exports a namespace with the same name as the module. Example: a function called xxx::aaa() most probably exists in a file inside a directory named "../xxx"
"""
}, {
  "role": "user",
  "content": "List only the source code for the function called in the line 22 of test.cpp"
}]

while True:
  data = json.dumps({
    "model": "gpt-4o-mini",
    "messages": messages,
    "tools": [{
      "type": "function",
      "function": {
        "name": "cat_file",
        "description": "Retrieves the context of a file. Result is empty if file was not found.",
        "strict": True,
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "filename to be retrieved"
            },
            "line_start": {
              "type": "integer",
              "description": "First line to retrieve, starting from 1."
            },
            "line_end": {
              "type": "integer",
              "description": "Last line to retrieve, starting from 1. Pass -1 to indicate the last line."
            }
          },
          "required": ["path", "line_start", "line_end"],
          "additionalProperties": False
        }
      },
    }, { 
      "type": "function",
      "function": {
        "name": "list_files",
        "description": "List files in a given directory. The tool don't recurse and it only works with directories. Result is empty if directory does not exist.",
        "strict": True,
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Directory to list"
            }
          },
          "required": ["path"],
          "additionalProperties": False
        }
      },
    }, { 
      "type": "function",
      "function": {
        "name": "grep",
        "description": "Given a directory name and a text, it will search for files in that directory containing said text.",
        "strict": True,
        "parameters": {
          "type": "object",
          "properties": {
            "directory": {
              "type": "string",
              "description": "directory name"
            },
            "text": {
              "type": "string",
              "description": "text to search"
            }
          },
          "required": ["directory", "text"],
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
  
  cho = json.loads(resp_data)['choices'][0]
  msg = cho['message']
  
  class Funcs:
    def cat_file(path, line_start, line_end):
      print(f'cat {path} {line_start} {line_end}')
      try:
        with open(path, 'r') as f:
          return f.readlines()[line_start:line_end]
      except FileNotFoundError:
        return ''
      
    def list_files(path):
      print(f'listing {path}')
      try:
        return os.listdir(path)
      except NotADirectoryError:
        return []
      except FileNotFoundError:
        return []

    def grep(directory, text):
      print(f'grep {directory} {text}')
      try:
        p = subprocess.run(['rg',text,directory],stdout=subprocess.PIPE,text=True,check=True)
        if (p.stdout == None): return []
        return p.stdout.splitlines()
      except subprocess.CalledProcessError:
        return []

    def find_module(module):
      print(f'find module {module}')
      return f'../{module}/{module}.cppm'
  
  if msg['content']:
    print(msg['content'])

  if (cho['finish_reason'] == 'stop'):
    break
  
  messages.append(msg)
  for tc in msg['tool_calls']:
    fn = tc['function']
    args = json.loads(fn['arguments'])
    messages.append({
      "role": "tool",
      "tool_call_id": tc["id"],
      "content": str(getattr(Funcs, fn['name'])(**args))
    })

