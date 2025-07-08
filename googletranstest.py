from google.cloud import translate
from google.oauth2 import service_account

def translate_text(text="Hello, world!", project_id="boreal-rain-465311-e2"):
    credentials = service_account.Credentials.from_service_account_file(
        'D:/HonoursProject/keys/boreal-rain-465311-e2-65805b7c44c1.json'
    )
    client = translate.TranslationServiceClient(credentials=credentials)
    
    location = "global"
    parent = f"projects/{project_id}/locations/{location}"
    response = client.translate_text(
        request={
            "parent": parent,
            "contents": [text],
            "mime_type": "text/plain",
            "source_language_code": "en-US",
            "target_language_code": "es",
        }
    )

    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))

translate_text()