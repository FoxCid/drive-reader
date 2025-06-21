{\rtf1\ansi\ansicpg1252\cocoartf2820
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11905\paperh16837\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx566\tx1133\tx1700\tx2267\tx2834\tx3401\tx3968\tx4535\tx5102\tx5669\tx6236\tx6803\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # Drive Reader Flask app - s\'e9curis\'e9 et pr\'eat pour Render\
import os\
from flask import Flask, jsonify\
from google.oauth2.credentials import Credentials\
from googleapiclient.discovery import build, errors\
from googleapiclient.http import MediaIoBaseDownload\
import io\
import pandas as pd\
\
app = Flask(__name__)\
\
ACCESS_TOKEN = os.environ.get('ACCESS_TOKEN')\
REFRESH_TOKEN = os.environ.get('REFRESH_TOKEN')\
CLIENT_ID = os.environ.get('CLIENT_ID')\
CLIENT_SECRET = os.environ.get('CLIENT_SECRET')\
TOKEN_URI = 'https://oauth2.googleapis.com/token'\
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']\
\
def get_service():\
    creds = Credentials(\
        token=ACCESS_TOKEN,\
        refresh_token=REFRESH_TOKEN,\
        token_uri=TOKEN_URI,\
        client_id=CLIENT_ID,\
        client_secret=CLIENT_SECRET,\
        scopes=SCOPES\
    )\
    return build('drive', 'v3', credentials=creds)\
\
@app.route('/list-files', methods=['GET'])\
def list_files():\
    service = get_service()\
    results = service.files().list(pageSize=10, fields="files(id, name, mimeType)").execute()\
    items = results.get('files', [])\
    return jsonify(\{'files': items\})\
\
@app.route('/read-excel/<file_id>', methods=['GET'])\
def read_excel(file_id):\
    service = get_service()\
    try:\
        request = service.files().get_media(fileId=file_id)\
        fh = io.BytesIO()\
        downloader = MediaIoBaseDownload(fh, request)\
        done = False\
        while not done:\
            status, done = downloader.next_chunk()\
        fh.seek(0)\
        df = pd.read_excel(fh)\
        return jsonify(df.to_dict(orient='records'))\
    except errors.HttpError as error:\
        return jsonify(\{'error': str(error)\}), 500\
\
if __name__ == '__main__':\
    app.run(debug=True)}