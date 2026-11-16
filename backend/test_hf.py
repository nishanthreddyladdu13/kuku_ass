"""Standalone Stability AI Stable Image Core API test.

Run from your backend directory after setting STABILITY_API_KEY in .env:
    python test_stability.py
"""
from pathlib import Path
import os
import sys

import httpx
from dotenv import load_dotenv

HERE = Path(__file__).resolve().parent
load_dotenv(HERE / '.env')

api_key = os.getenv('STABILITY_API_KEY', '').strip()
if not api_key:
    sys.exit('Missing STABILITY_API_KEY in backend/.env')

url = 'https://api.stability.ai/v2beta/stable-image/generate/core'
prompt = (
    'Photorealistic cinematic film still, medium-wide eye-level camera shot. '
    'A woman named Maya stands beside a wooden table in a living room. '
    'A man named Arun sits in a chair facing her. '
    'Warm practical lighting, realistic proportions, dramatic film composition, '
    'one single frame, no captions or text.'
)

print('Requesting one image from Stability AI Stable Image Core (3 credits)...')
try:
    response = httpx.post(
        url,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Accept': 'image/*',
        },
        # httpx encodes fields as multipart/form-data using (None, value).
        files={
            'prompt': (None, prompt),
            'output_format': (None, 'png'),
            'aspect_ratio': (None, '16:9'),
        },
        timeout=120,
    )
except httpx.RequestError as exc:
    sys.exit(f'Connection failed: {type(exc).__name__}: {exc}')

if response.status_code != 200:
    print(f'API error: HTTP {response.status_code}')
    try:
        details = response.json()
        print('Details:', details.get('errors') or details.get('message') or details)
    except ValueError:
        print('Details:', response.text[:1000])
    if response.status_code == 402:
        print('Your Stability account needs additional credits; check the billing dashboard.')
    elif response.status_code == 401:
        print('Check that STABILITY_API_KEY is valid.')
    sys.exit(1)

content_type = response.headers.get('content-type', '')
if not content_type.startswith('image/'):
    sys.exit(f'Unexpected response type: {content_type}. Body: {response.text[:500]}')

output_path = HERE / 'test_stability.png'
output_path.write_bytes(response.content)
print(f'Success! Image saved to: {output_path}')
print(f'Image size: {len(response.content):,} bytes')
