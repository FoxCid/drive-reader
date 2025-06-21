import os
import io
import json
import pandas as pd
from flask import Flask, jsonify
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

app = Flask(__name__)

SERVICE_ACCOUNT_INFO = json.loads(os.environ.get("GOOGLE_CREDENTIALS_JSON"))
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

credentials = service_account.Credentials.from_service_account_info(
    SERVICE_ACCOUNT_INFO, scopes=SCOPES
)

def get_service():
    return build('drive', 'v3', credentials=credentials)

@app.route('/')
def index():
    return "Service Drive Reader en ligne."

@app.route('/list-files', methods=['GET'])
def list_files():
    service = get_service()
    results = service.files().list(pageSize=10, fields="files(id, name, mimeType)").execute()
    return jsonify({'files': results.get('files', [])})

@app.route('/read-excel/<file_id>', methods=['GET'])
def read_excel(file_id):
    try:
        service = get_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        df = pd.read_excel(fh)
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
