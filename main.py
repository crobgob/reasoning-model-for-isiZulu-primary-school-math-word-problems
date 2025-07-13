#Program that creates a parquet dataframe for GSM8K translated to Zulu. We will have three columns: question, CoT and answer.
#Corey Webb, Bsc Hons 2025, UCT
#10/07/2025

import pandas as pd
from google.cloud import translate
from google.oauth2 import service_account

def translate_to_zul(text="Hello, world!", project_id="boreal-rain-465311-e2"):
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
            "target_language_code": "zu",
        }
    )

    for translation in response.translations:
        print("Translated text: {}".format(translation.translated_text))

    return translation.translated_text

#Don't use translated cot bc could be wrong
#All cot answers end in ###{answer}
def extract_ans(cot_answer):
    split = cot_answer.split("####")
    #Checks that there is one and only one answer
    if len(split) == 2:
        ans = int(split[1].replace(",", "").strip())
        return ans
    else:
        print("Expected one answer, found {}".format(len(split) - 1))


def main():
    df = pd.read_parquet("hf://datasets/openai/gsm8k/main").head(10)

    questions_zul = []
    cot_zul = []
    int_answers = []

    for row in range(len(df)):
        print("Row: ", row)
        question = df.iloc[row]['question']
        answer = df.iloc[row]['answer']
        trans_quest = translate_to_zul(question)
        trans_cot = translate_to_zul(answer)
        int_ans = extract_ans(answer)

        questions_zul.append(trans_quest)
        cot_zul.append(trans_cot)
        int_answers.append(int_ans)

    #Create final dataframe
    zulu_df = pd.DataFrame({
        "question": questions_zul,
        "CoT": cot_zul,
        "answer": int_answers
    })

    #Create parquet file
    parquet_path = "GSM8K_Zulu.parquet"
    zulu_df.to_parquet(parquet_path, index=False)

    
if __name__ == "__main__":
    main()