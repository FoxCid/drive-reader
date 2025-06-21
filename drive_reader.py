 # Drive Reader Flask app - sécurisé et prêt pour Render
import os
from flask import Flask, jsonify
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import io
import pandas as pd

app = Flask(__name__)

# Chargement des variables sensibles depuis les variables d'environnement
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')
REFRESH_TOKEN = os.environ.get('REFRESH_TOKEN')
CLIENT_ID = os.environ.get('CLIENT_ID')
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')
TOKEN_URI = 'https://oauth2.googleapis.com/token'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def get_service():
    creds = Credentials(
        token=ACCESS_TOKEN,
        refresh_token=REFRESH_TOKEN,
        token_uri=TOKEN_URI,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES
    )
    return build('drive', 'v3', credentials=creds)

@app.route('/list-files', methods=['GET'])
def list_files():
    try:
        service = get_service()
        results = service.files().list(pageSize=10, fields="files(id, name, mimeType)").execute()
        items = results.get('files', [])
        return jsonify({'files': items})
    except HttpError as error:
        return jsonify({'error': str(error)}), 500

@app.route('/read-excel/<file_id>', methods=['GET'])
def read_excel(file_id):
    try:
        service = get_service()
        request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
        fh.seek(0)
        df = pd.read_excel(fh)
        return jsonify(df.to_dict(orient='records'))
    except HttpError as error:
        return jsonify({'error': str(error)}), 500

if __name__ == '__main__':
    app.run(debug=True)
