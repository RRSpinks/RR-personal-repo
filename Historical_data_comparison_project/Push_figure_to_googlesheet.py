import matplotlib.pyplot as plt
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Generate a scatterplot and save as a PNG image
np.random.seed(0)
x = np.random.rand(50)
y = np.random.rand(50)
colors = np.random.rand(50)
area = (30 * np.random.rand(50))**2
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.savefig('scatterplot.png')

# Set up Google API credentials
SERVICE_ACCOUNT_FILE = 'path/to/your_service_account_key.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']

# Authenticate with Google API
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, SCOPES)
sheets_api = build('sheets', 'v4', credentials=credentials)
drive_api = build('drive', 'v3', credentials=credentials)

# Upload the scatterplot image to Google Drive
file_metadata = {'name': 'scatterplot.png', 'mimeType': 'image/png'}
media = MediaFileUpload('scatterplot.png', mimetype='image/png')
file = drive_api.files().create(body=file_metadata, media_body=media, fields='id').execute()

# Define the Google Sheet ID and the location to insert the image
SHEET_ID = 'your_google_sheet_id'
sheet_location = 'A1'

# Insert the scatterplot image to Google Sheet
image_id = 'scatterplot_image'
insert_image_request = {
    'addImage': {
        'objectId': image_id,
        'url': f'https://drive.google.com/uc?id={file.get("id")}',
        'insertDimension': {
            'location': {
                'sheetId': 0,
                'dimension': 'ROWS',
                'startIndex': 0,
                'endIndex': 1
            },
            'inheritFromBefore': False
        },
        'overlayPosition': {
            'anchorCell': {
                'sheetId': 0,
                'rowIndex': 0,
                'columnIndex': 0
            },
            'offsetXPixels': 0,
            'offsetYPixels': 0
        }
    }
}

sheets_api.spreadsheets().batchUpdate(spreadsheetId=SHEET_ID, body={'requests': [insert_image_request]}).execute()
