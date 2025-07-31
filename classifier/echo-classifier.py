def main():

    import os
    import subprocess

    packages_location = os.getenv("PYTHONPATH", "/app/packages")
    os.environ["TORCH_HOME"] = "/app/models/whisper_model/torch"

    # Install everything in vendor folder by passing filenames to pip install
    packages = [os.path.join(packages_location, f) for f in os.listdir(packages_location)if f.endswith(".whl") or f.endswith(".tar.gz")]

    subprocess.run(["pip", "install", "--no-index", "--find-links", packages_location] + packages, check=True)

    import psycopg2
    from loguru import logger
    import re
    from tqdm import tqdm
    import time
    import sys
    import spacy
    from better_profanity import profanity
    from transformers import AutoTokenizer, AutoModelForSequenceClassification,pipeline

    nlp = spacy.load("/app/models/en_core_web_sm/en_core_web_sm-3.8.0")

    # load the Mistral 7b model, languauge model, and tokenizer
    #C:\Users\HICKMAN\GoDAP_Echo\mistral-7b-instruct-v0.1.Q4_K_M.gguf
    #app\models\mistral_model\mistral-7b-instruct-v0.1.Q4_K_M.gguf
    logger.info(f"Custom Engineering Classifier loading...")
    start_time = time.time()
    model_path = "/app/models/classifier_model/engineering_classifierV7"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    end_time = time.time() - start_time
    logger.info(f"Classifier model loaded in {end_time} seconds.")

    profanity.load_censor_words()

    #------------ connecting to the DB -------------
    DB_HOST = os.getenv("PGHOST")
    DB_NAME = os.getenv("PGNAME")
    DB_USER = os.getenv("PGUSER")
    DB_PASSWORD = os.getenv("PGPASSWORD")
    DB_PORT = int(os.getenv("PGPORT"))

    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        cur = conn.cursor()
        logger.info("Database connection established successfully.")
        logger.info(f"Connected to database: {DB_NAME} at {DB_HOST}:{DB_PORT}")
    except Exception as e:
        logger.info("Failed to connect to the database.")
        logger.info(f"Error: {e}")
    #------------------------------------------------------------------

    #------------------ summarzing the transcriptions ------------------
    #inference_tags = []
    #inference_filepaths = []
    count = 0
    try:
        while True:
            print(count, "iterations completed.")
            if count >= 1:
                logger.info("Going to sleep for 60 minutes...")
                time.sleep(3600)
            cur.execute("SELECT triggerid FROM data.trigger WHERE transcription_text != '' AND transcription_text is NOT NULL")
            master_list_triggers = [row[0] for row in cur.fetchall()]

            cur.execute("SELECT triggerid FROM data.trigger WHERE cleaned_transcription is NOT NULL AND cleaned_transcription != ''")
            processed_triggers = [row[0] for row in cur.fetchall()]

            unprocessed_triggers = [id for id in master_list_triggers if id not in processed_triggers]
            print(len(unprocessed_triggers), "uncleaned transcriptions found.")
        
            for id in tqdm(unprocessed_triggers, desc="Cleaning Transcriptions", unit="transcription"):# for loop to detect tags in the transcription_text and then use those tags to select the correct document for context
            
                tqdm.write(f"Now cleaning transcription for trigger {id}") 
                sys.stdout.flush() 
                cur.execute("SELECT transcription_text FROM data.trigger WHERE triggerid = %s",(id,) )
                result = cur.fetchone()
                if result:
                    text = result[0]
                
                def redact_names(text):#spacy library to remove names
                    doc = nlp(text) # processes name through model, labels names, dates, etc
                    for ent in doc.ents:
                        if ent.label_ in ["PERSON"]: # if a word has the person label
                            text = text.replace(ent.text, "Driver") # replaces the text with Driver 
                    return text

                def censor_profanity(text):#better_profanity library to censor
                    return profanity.censor(text)# censor any bad text

                tqdm.write(f"Pre-processing text: {text}")
                sys.stdout.flush()

                text = redact_names(text) # take out names
                text = censor_profanity(text) # take out profanity

                cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_schema = 'data' AND table_name = 'tags' AND data_type = 'bit'""")
                extracted_columns = [row[0] for row in cur.fetchall()]
                keywords = [col.replace("_", " ") for col in extracted_columns] 
                

                def contains_keyword(text, keywords):
                    text_lower = text.lower()
                    return any(keyword.lower() in text_lower for keyword in keywords) #return true if any keyword in text_lower

            # Clean transcription with classifier model
                def classifier_cleaning(text):
                    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
                    result = classifier(text)
                    return result

                if contains_keyword(text, keywords):
                    cleaned = text # No LLM needed, valid engineering tag
                else:
                    result = classifier_cleaning(text)
                    label = result[0]['label']
                    score = result[0]['score']
                    if label == "LABEL_1" and score > 0.6:
                        cleaned = text
                    else:
                        cleaned = "No engineering content. Review audio file manually."
                
                tqdm.write(f"Post Processing text: {cleaned}")
                sys.stdout.flush()
                try:
                    tqdm.write(f"Attempting to commit for {id} ")
                    sys.stdout.flush()
                    cur.execute("UPDATE data.trigger SET cleaned_transcription = %s WHERE triggerid = %s", (cleaned,id))
                    conn.commit()
                    tqdm.write(f"Cleaned transcription committed for {id}")
                    sys.stdout.flush()
                except Exception as e:
                    tqdm.write("Failed to commit.")
                    tqdm.write(f"Error: {e}")
                    sys.stdout.flush()
                    conn.rollback()
            count += 1

    except KeyboardInterrupt:
        logger.info("Shutting down.")
        sys.stdout.flush()
            #-----------------------------------------------------------------
    
if __name__ == "__main__":
    main()
