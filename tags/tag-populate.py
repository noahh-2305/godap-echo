def main():
    import librosa
    import numpy as np
    import os
    import psycopg2

    from loguru import logger
    from tqdm import tqdm
    import time
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
    #---------------------------------------------

    #------------- OAW Tagging -------------------
    def OAWDetection(wav_path):
        # Sound Detection for Optical Acoustic Warnings (OAW)
        y, sr = librosa.load(wav_path) # returns audio time series (y, NumPy array of sample amplitudes) and sampling rate (sr, rate of measurements per second)

        STFT=librosa.stft(y) # creating a short-time fourier transform 
        spectro_dB=librosa.amplitude_to_db(abs(STFT), ref=np.max)# converts amplitude spectrogram to dB-scaled spectrogram 

        freqs = librosa.fft_frequencies(sr=sr) # taking the frequencies from the STFT and making a boolean mask for the OAW sound detection
        beep_band = (freqs >= 5500) & (freqs <= 8192)

        spectro_band = spectro_dB[beep_band, :] # selects specific rows from the spectrogram using the boolean mask

        band_energy = np.mean(spectro_band, axis=0) # outputs the mean energy (dBs) across that masked spectro band, to tell if and when the sound is occuring

        threshold = -75 # threshold for the beep noise in dB
        frame_count_threshold = 2 

        beep_frames = band_energy > threshold

        count2 = 0
        beep_detected = False
        for hit in beep_frames: # 'hit' is when a beep_frame is true, meaning the dB is above the threshold for the specified frequency range
            if hit:
                count2 += 1
                if count2 > frame_count_threshold:
                    beep_detected = True
                    break
            else:
                count2 = 0

        OAW = '0'
        if beep_detected:
            OAW = '1'
        return OAW
    #------------------------------------------------------
    count=0
    try:
        while True:
            if count >= 1:
                time.sleep(3600)

            keep_alive_time = time.time()  # Reset the keep-awake timer
            if conn.closed:
                    conn = psycopg2.connect(
                    host=DB_HOST,
                    database=DB_NAME,
                    user=DB_USER,
                    password=DB_PASSWORD,
                    port=DB_PORT)         # if the SQL connection is closed, reconnect
                    cur = conn.cursor()
            if time.time() - keep_alive_time > 300:  
                conn.set_session
                conn.cursor().execute("SET idle_in_transaction_session_timeout = 0;")  # make sure that psycopg2 doesn't timeout
                keep_alive_time = time.time()
            

            try:
                cur.execute("""SELECT column_name FROM information_schema.columns WHERE table_schema = 'data' AND table_name = 'tags' AND data_type = 'bit'""")
                all_columns = [row[0] for row in cur.fetchall()]

                cur.execute("""SELECT * FROM data.tags LIMIT 1""")
                sample_row = cur.fetchone()

                col_names = [desc[0] for desc in cur.description]

                existing_cols = [col for col in all_columns if col in col_names and sample_row[col_names.index(col)] is not None]

                new_cols_added = set(all_columns) - set(existing_cols)

                rows = []
                if new_cols_added:
                    logger.info("New tag found in table. Retagging all triggers.")
                    cur.execute("SELECT triggerid, trigger_filepath, wav_filename, transcription_text FROM data.trigger WHERE transcription_text IS NOT NULL")
                    rows = cur.fetchall()
                    rows = [(triggerid, trigger_path.replace("\\\\stnafcvdo030.us164.corpintra.net\\ntm", "/data030").replace("\\", "/") if trigger_path.startswith("\\\\stnafcvdo030.us164.corpintra.net\\ntm") else trigger_path, wav_path, transcription_text) for triggerid, trigger_path, wav_path, transcription_text in rows]
                else:
                    logger.info("No new tags. Will now check if anything in database is untagged.")
                    cur.execute("SELECT triggerid FROM data.tags WHERE tagging_status = 'untagged'")
                    untagged_ids = [row[0] for row in cur.fetchall()]

                    for untagged_id in untagged_ids:
                        cur.execute(f"""SELECT triggerid, trigger_filepath, wav_filename, transcription_text FROM data.trigger WHERE triggerid = %s AND transcription_text IS NOT NULL""", (untagged_id,))

                        result = cur.fetchone()
                        if result:
                            triggerid, trigger_path, wav_path, transcription_text = result
                            if trigger_path.startswith("\\\\stnafcvdo030.us164.corpintra.net\\ntm"):
                                trigger_path = trigger_path.replace("\\\\stnafcvdo030.us164.corpintra.net\\ntm", "/data030").replace("\\", "/")
                            rows.append((triggerid, trigger_path, wav_path, transcription_text))
                
                columns = all_columns
                
                for triggerid, trigger_path, wav_path, transcription_text in tqdm(rows, desc="Tagging triggers", unit="trigger"):
                    found_keywords = {}
                    for col in columns:
                        if col.lower() in transcription_text.lower():
                            found_keywords[col] = '1'
                        elif col.lower() == "oaw" and os.path.join(trigger_path,wav_path).lower().endswith(".wav"):
                            found_keywords[col] = OAWDetection(os.path.join(trigger_path,wav_path)) 
                        else:
                            found_keywords[col] = '0'

                    set_keywords = ",".join([f'"{key}" = %s' for key in found_keywords]) # ["OAW = %s, "keyword = %s, etc"]
                    values = list(found_keywords.values())
                    values.append("tagged")
                    values.append(triggerid)

                    try:
                        cur.execute(f"UPDATE data.tags SET {set_keywords}, tagging_status = %s WHERE triggerid= %s", values)
                        conn.commit()
                        logger.info(f"data.tags commit successful for trigger_id: {triggerid}.")
                    except Exception as e:
                        tqdm.write("Commit failed:", e)
                        conn.rollback()
                        cur.execute("UPDATE data.tags SET tagging_status = %s WHERE triggerid = %s", ("error while tagging", triggerid))
                        conn.commit()
            except Exception as e:
                logger.info("Error during tagging process.")
                logger.info(f"Error: {e}")
            logger.info("Tagging cycle complete. Sleeping for 60 minutes.")
            count+=1

    except KeyboardInterrupt:
        logger.info("Shutting down.")
    cur.close()
    conn.close()

    print('Tagging Complete.')

if __name__ == "__main__":
        main()