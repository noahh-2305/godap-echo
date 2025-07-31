def main():
    
    import os
    import subprocess

    packages_location = os.getenv("PYTHONPATH", "/app/packages")
    os.environ["TORCH_HOME"] = "/app/models/whisper_model/torch"

    # Install everything in vendor folder by passing filenames to pip install
    packages = [os.path.join(packages_location, f) for f in os.listdir(packages_location)if f.endswith(".whl") or f.endswith(".tar.gz")]

    subprocess.run(["pip", "install", "--no-index", "--find-links", packages_location] + packages, check=True)

    import psycopg2
    from pathlib import Path
    import sys
    import logging

    import whisper
    import whisperx
    import whisperx.alignment as alignment
    import torch

    import numpy as np
    import time

    from dotenv import load_dotenv

    from loguru import logger
    from tqdm import tqdm
    import re
    from datetime import datetime, timezone, timedelta
    os.environ["DBUSER"] = os.getenv("PGNAME")
    os.environ["DBPASS"] = os.getenv("PGPASSWORD")
    from godap import SignalData, Signal
    import pandas as pd
    from shapely.geometry import Polygon
    import geopandas

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, force=True)

    load_dotenv(Path(__file__).resolve().parent / ".env") # IF RUNNING LOCALLY, MAKE IT PARENT PARENT

    #----------------- Patching the broken get_wildcard_emission ---------------------
    def get_wildcard_emission_fixed(frame_emission, tokens, blank_id): # has to do with tokens.clamp returning a tensor of data type that was not torch.int, torch.long, or torch.bool
    # this function will always return at least a 1D tensor
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = tokens.long()

        regular_scores = frame_emission[tokens.clamp(min=0)]
        mask = tokens != blank_id
        avg_score = regular_scores[mask].mean() if mask.any() else torch.tensor([0.0])  # make it 1D
        return avg_score.unsqueeze(0) if avg_score.dim() == 0 else avg_score


    alignment.get_wildcard_emission = get_wildcard_emission_fixed
    #----------------------------------------------------------------------------------

    #------------ connecting to the DB -------------
    DB_HOST = os.getenv("PGHOST")
    DB_NAME = os.getenv("PGNAME")
    DB_USER = os.getenv("PGUSER")
    DB_PASSWORD = os.getenv("PGPASSWORD")
    DB_PORT = os.getenv("PGPORT")

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
       sys.exit(1)
    #---------------------------------------------
     
    #------------ transcription model setup --------------------
    
    logger.info("Reached model loading section.")
     

    try:
        #model_path = r"F:\GoDAP_Echo_Model\models\whisper_model\whisper\medium.pt"
        model_path = "/app/models/whisper_model/whisper/large-v3.pt"  # Path to the Whisper model, only useable with portainer
        begintime = time.time()
        model = whisper.load_model(model_path, device="cpu")
        elapsed = time.time()-begintime
        #model = whisper.load_model("medium", device="cpu")
        logger.info(f"Whisper Model Loaded in {elapsed:.2f} seconds.")
         

    except Exception as e:
        logger.info("Failed to load transcription/diarization models.")
        logger.info(f"Error: {e}")
         

        sys.exit(1)
    #-----------------------------------------------------------

    count = 0
    try:
        while True:
            if count >= 1:
                logger.info("Going to sleep for 60 minutes...")
                time.sleep(3600)
            #-------------use the databucket ID to grab all unprocessed trigger-packages------------
            logger.info("Pulling buckets...")
             
            cur.execute("SELECT id FROM data.file WHERE fk_filetype_id_file = 4 ORDER BY id DESC")
            master_list_databuckets = [row[0] for row in cur.fetchall()]

            cur.execute("SELECT DISTINCT databucketid FROM data.trigger_jobqueue")
            processed_databuckets = [row[0] for row in cur.fetchall()]

            unprocessed_databuckets = [id for id in master_list_databuckets if id not in processed_databuckets]
            print(len(unprocessed_databuckets), "unprocessed databuckets found.")
             
            keep_alive_time = time.time()
            logger.info("Processing starting now...")
            #----------------------------------------------------------------------------------------


            #------------- main loop for processing databuckets -------------------
            for id in tqdm(unprocessed_databuckets, desc="Processing Databuckets", unit="bucket"):
                tqdm.write(f"Now on databucket {id}")
                sys.stdout.flush()
        
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


                # Check if the databucket is already in the job queue
                cur.execute("SELECT databucketid from data.trigger_jobqueue WHERE databucketid = %s", (id,))
                rows = cur.fetchall()

                if rows == []:
                    cur.execute("INSERT INTO data.trigger_jobqueue (databucketid, jobstatus) VALUES (%s,%s)",(id,"currently being processed..."))
                    conn.commit()
                else:
                    cur.execute("UPDATE data.trigger_jobqueue SET jobstatus = %s WHERE databucketid =%s ",("currently being processed...", id))
                    conn.commit()
            
                databucket_id = id
                databucket_filepath = ""

                # grab the file path column from data.file table
                def grab_databucket_filepath():
                    cur.execute(f"SELECT filepath from data.file WHERE id = %s", (databucket_id,))
                    databucket_filepath = cur.fetchall()[0][0]
                    #----------- this section needed if using mounted drive on docker ------------
                    
                    if databucket_filepath.startswith(r"\\stnafcvdo030.us164.corpintra.net\ntm"):
                        databucket_filepath = databucket_filepath.replace(r"\\stnafcvdo030.us164.corpintra.net\ntm", "/data030").replace("\\", "/")
                    
                    #---------------------------------------------------------------
                    
                    index = databucket_filepath.find(r'\GoNTM')
                    if index != -1:
                        databucket_filepath = databucket_filepath[:index]
                    return databucket_filepath

                # grab all the trigger-package folders in the trigger-packages folder
                def grab_trigger_packages(databucket_filepath):
                    filepath = f"{databucket_filepath}/trigger-packages"
                    trigger_packages_temp = [item for item in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, item))]
                    if not trigger_packages_temp:
                        #smart_print("No Files in the Trigger-Packages folder for this data bucket")
                        trigger_packages = []
                    else:
                        #print(f"Trigger Package Files present in Databucket {databucket_id}")
                        trigger_packages = trigger_packages_temp
                    return trigger_packages
                #---------------------------------------------------------------------------------


                #------------- grabbing the files from the network drive--------------
                def trigger_package_file_puller(trigger_package_path):
                    # main function for grabbing files names

                    files = os.listdir(trigger_package_path)
                    wav_file = ""
                    hostcam_jpgs=[]
                    hostcam_mp4s=[]
                    mf4_file=""

                    def get_wav():
                        # grabs the singular .wav file
                        nonlocal wav_file,files
                        for file in files[:]:
                            if file.endswith(".wav"):
                                wav_file = file
                                files.remove(file)
                                break
                    
                    def get_mf4():
                        # grabs the singular .MF4 file
                        nonlocal mf4_file, files
                        for file in files[:]:
                            if file.endswith(".MF4"):
                                mf4_file = file
                                files.remove(file)
                                break
                        
                    
                    def get_hostcam_mp4():
                        # grabs all hostcam .mp4 files
                        nonlocal hostcam_mp4s, files
                        for file in files[:]:
                            if file.endswith(".mp4"):
                                hostcam_mp4s.append(file)
                                files.remove(file)
                        return
                    
                    def get_hostcam_jpg():
                        # grabs all hostcam .jpg files
                        nonlocal hostcam_jpgs, files
                        for file in files[:]:
                            if file.endswith(".jpg"):
                                hostcam_jpgs.append(file)
                                files.remove(file)
                        return
                    
                    
                    if any(file.endswith(".wav") for file in files):
                        get_wav()
                    if any(file.endswith(".MF4") for file in files):
                        get_mf4()
                    if any(file.endswith(".jpg") for file in files):
                        get_hostcam_jpg()
                    if any(file.endswith(".mp4") for file in files):
                        get_hostcam_mp4()

                    pulled_media_files = {".wav" : wav_file,".mf4": mf4_file,".mp4s":hostcam_mp4s,".jpgs":hostcam_jpgs}
                    return pulled_media_files
                #----------------------------------------------------------------

                #-------------- running code ------------------------------------
                tqdm.write("Grabbing databucket filepath...")
                sys.stdout.flush()
                databucket_filepath = grab_databucket_filepath()
                trigger_check_path = f"{databucket_filepath}/trigger-packages"
                
                tqdm.write("Checking for trigger packages in databucket...")
                sys.stdout.flush()
                # Check if the folder exists and is not empty
                if not os.path.exists(trigger_check_path) or not os.listdir(trigger_check_path):
                    tqdm.write(f"No Trigger-Packages folder found or it's empty for this Databucket. Moving to next Databucket.")
                    sys.stdout.flush()
                    cur.execute("UPDATE data.trigger_jobqueue SET jobstatus = %s WHERE databucketid =%s ",("databucket processed; contains no triggers", databucket_id))
                    conn.commit()
                    tqdm.write(f"Databucket {databucket_id}  marked in job queue as having no triggers, moving to next bucket...")
                    sys.stdout.flush()
                    continue
                tqdm.write(f"Trigger packages exist for this databucket.")
                sys.stdout.flush()

                trigger_packages = grab_trigger_packages(databucket_filepath)
                
                pulled_files_from_trigger_folder = {}

                # -------------- going through each trigger package folder and pulling files ----------------
                for trigger_package in trigger_packages:
                    package_filepath = f"{databucket_filepath}/trigger-packages/{trigger_package}"

                    cur.execute("SELECT triggerid, databucketid FROM data.trigger WHERE trigger_filepath = %s", (package_filepath,))
                    temp = cur.fetchone() # returns a single tuple
                    if temp:
                        tid, dbid = temp
                        tqdm.write(f"Trigger filepath {package_filepath} already exists in data.trigger table for existing Databucket {dbid}. It is trying to have a duplicate trigger_file path with current Databucket {databucket_id}. Skipping.")
                        sys.stdout.flush()
                        continue  # Skip this trigger_package if it the filepath was already previously written to table

                    pulled_files_from_trigger_folder[trigger_package] = trigger_package_file_puller(package_filepath)

                    mp4s = {f"hostcam{i}_mp4" : "" for i in range(1,9)}
                    if pulled_files_from_trigger_folder[trigger_package][".mp4s"]:
                        for mp4 in pulled_files_from_trigger_folder[trigger_package].get(".mp4s",[]):
                            for i in range (1,9):
                                if f'HOSTCAM{i}' in mp4:
                                    mp4s[f"hostcam{i}_mp4"] = mp4
                                    break
                    
                    jpgs = {f"hostcam{i}_jpg" : "" for i in range(1,9)}
                    if pulled_files_from_trigger_folder[trigger_package][".jpgs"]:
                        for jpg in pulled_files_from_trigger_folder[trigger_package].get(".jpgs",[]):
                            for i in range (1,9):
                                if f'HOSTCAM{i}' in jpg:
                                    jpgs[f"hostcam{i}_jpg"] = jpg
                                    break
                    else:
                        pulled_files_from_trigger_folder[trigger_package][".jpgs"] = ["","","","","","","",""]

                    #----------------- Transcription and Diarization -------------------
                    transcribed_text = "no translatable words"
                    if pulled_files_from_trigger_folder[trigger_package][".wav"]:
                        # Initial, Fast transcription with regular whisper (can be on GPU if available)
                        no_speech_prob = 1.0
                        wav_path = f"{package_filepath}/{pulled_files_from_trigger_folder[trigger_package][".wav"]}"
                        tqdm.write("Now attempting to transcribe text...")
                        sys.stdout.flush()
                        result = model.transcribe(wav_path, language="en", fp16 = False)
                        if result.get("segments"):
                            no_speech_prob = result["segments"][0].get("no_speech_prob", 0)
                        if no_speech_prob > 0.6:
                            # if the .wav is just background noise, no words, this is preventing the library from writing placeholder words
                            tqdm.write("No words present in WAV file.")
                            sys.stdout.flush()
                        else:
                            transcribed_text = result["text"]
                            # Alignment on CPU with whisperx (basically matches the transcribed words with the audio file, mainly used for training later on)
                            audio = whisperx.load_audio(wav_path)
                            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device="cpu") # returns the alignment model object and metadata (phoneme mappings, language-specific settings)
                            aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device="cpu", return_char_alignments=False)
                            # Assign speakers to words
                            
                            transcribed_text = " ".join(segment["text"].strip() for segment in aligned_result["segments"])
                            if transcribed_text:
                                tqdm.write("Transcription Successful.")
                                sys.stdout.flush()
                    else:
                        pulled_files_from_trigger_folder[trigger_package][".wav"] = ""
                        tqdm.write(f"Trigger does not have a .wav file associated with it.")
                        sys.stdout.flush()
                    #-----------------------------------------------------------------------

                    if not pulled_files_from_trigger_folder[trigger_package][".mf4"]:
                        pulled_files_from_trigger_folder[trigger_package][".mf4"] = ""
                        tqdm.write(f"Trigger does not have a .mf4 file associated with it.")
                        sys.stdout.flush()
                
                    #----------------- Extracting info from filepath name -------------------
                    tqdm.write("Now attempting to extract info from filepath name")
                    sys.stdout.flush()
                    timestamp_obj = None
                    formatted_timestamp = None
                    try:
                        matches = re.findall(r'(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', package_filepath)
                        if matches:
                            timestamp_str = matches[-1]
                            timestamp_obj = datetime.strptime(timestamp_str, '%Y-%m-%d_%H-%M-%S')# str to datetime object
                            timestamp_obj = timestamp_obj.replace(tzinfo=timezone.utc)

                            formatted_timestamp = timestamp_obj.strftime('%Y-%m-%d %H:%M:%S.%f %z')
                            formatted_timestamp = formatted_timestamp[:-8] + formatted_timestamp[-8:-5] + formatted_timestamp[-5:]
                    except Exception as e:
                        tqdm.write("Failed to extract timestamp from filepath")
                        tqdm.write(f"Error: {e}")
                        sys.stdout.flush()
                        timestamp_obj = None
                        formatted_timestamp = None

                    
                    truck_name = None
                    try:
                        match = re.search(r"[\\/]{1}([A-Za-z]{1,2}\d{4,5})[\\/]{1}log[\\/]", package_filepath)#raw string literal, only 1 occurance, both forward and backslash
                        if match:
                            truck_name = match.group(1)
                    except Exception as e:
                        tqdm.write("Failed to extract truck name from filepath")
                        tqdm.write(f"Error: {e}")
                        sys.stdout.flush()
                        truck_name = None
                    #----------------------------------------------------------

                    #----------------- Extracting location with GoDAP library -------------------
                    latitude = "n/a"
                    longitude = "n/a"
                    start_date = timestamp_obj-timedelta(seconds=5)
                    end_date = timestamp_obj
                    try:
                        if timestamp_obj and truck_name:
                            signal_list_backbone = [
                                Signal('Latitude_Cval_CTP','VP_CTP','Backbone_Int'),
                                Signal('Longitude_Cval_CTP','VP_CTP','Backbone_Int')
                            ]
                            signal_list_backbone

                            signal_data = SignalData(
                                signal_list_backbone,
                                physical_vehicle=truck_name,
                                window=[start_date, end_date],
                                fail_on_missing=True,
                            ) # makes dictionaries
                            lat_df = signal_data[Signal('Latitude_Cval_CTP','VP_CTP','Backbone_Int')]
                            lon_df = signal_data[Signal('Longitude_Cval_CTP','VP_CTP','Backbone_Int')]

                            #start_date = timestamp_obj-timedelta(seconds=5)
                            #end_date = timestamp_obj
                                # Extract latitude and longitude values (first row)
                            latitude = lat_df['Latitude_Cval_CTP'].iloc[0]
                            longitude = lon_df['Longitude_Cval_CTP'].iloc[0]

                            print(f"Latitude: {latitude}, Longitude: {longitude}")

                    except Exception as e:
                        print("Failed to grab coordinates")
                        print(e)

                    geom_swan_island = Polygon(zip([-122.71957, -122.72789, -122.70094, -122.69141], [45.57069, 45.56499, 45.55219, 45.55771]))
                    geom_corp3 = Polygon(zip([-122.71278, -122.71197, -122.71347, -122.71435], [45.56309, 45.56259, 45.56124, 45.56174]))
                    geom_corp12 = Polygon(zip([-122.70114, -122.70296, -122.70163, -122.70116, -122.70019], [45.55949, 45.55784, 45.55709, 45.55821, 45.55868]))
                    geom_madras = Polygon(zip([-121.17314, -121.17975, -121.15743, -121.15151, -121.15709], [44.67112, 44.66715, 44.65164, 44.65616, 44.66581]))
                    geom_smithers_winter_test = Polygon(zip([-84.82904, -84.79153, -84.79128, -84.82896], [46.35819, 46.35848, 46.34426, 46.34385]))
                    geom_portland_international_raceway = Polygon(zip([-122.70435, -122.69916, -122.69729, -122.69615, -122.6872, -122.6871, 122.6904, -122.68982], [45.59751, 45.60108, 45.59997, 45.60068, 45.59562, 45.59185, 45.59164, 45.59334]))
                    geom_fairbanks_alaska = Polygon(zip([-149.232554,  -149.103663, -146.547992, -146.019045], [64.276743, 65.406393, 65.277104, 64.189311]))

                    poly_swan_island = geopandas.GeoSeries(geom_swan_island)
                    poly_corp3 = geopandas.GeoSeries(geom_corp3)
                    poly_corp12 = geopandas.GeoSeries(geom_corp12)
                    poly_madras = geopandas.GeoSeries(geom_madras)
                    poly_smithers_winter_test = geopandas.GeoSeries(geom_smithers_winter_test)
                    poly_portland_international_raceway = geopandas.GeoSeries(geom_portland_international_raceway)
                    poly_fairbanks_alaska = geopandas.GeoSeries(geom_fairbanks_alaska)

                    def find_geo_location_name(lat, lon):
                        if lat is None and lon is None:
                            return []
                        offset = 0.001  

                        coords = [
                            (lon - offset, lat - offset),
                            (lon + offset, lat - offset),
                            (lon + offset, lat + offset),
                            (lon - offset, lat + offset),
                            (lon - offset, lat - offset) 
                        ]
                        geom_databucket = Polygon(coords)
                        poly_databucket = geopandas.GeoSeries(geom_databucket)
                        intersections = dict(
                            swan_island = poly_databucket.intersects(poly_swan_island)[0],
                            corp3_lot = poly_databucket.intersects(poly_corp3)[0],
                            corp12 = poly_databucket.intersects(poly_corp12)[0],
                            madras_hdpg = poly_databucket.intersects(poly_madras)[0],
                            smithers_winter_test = poly_databucket.intersects(poly_smithers_winter_test)[0],
                            portland_international_raceway = poly_databucket.intersects(poly_portland_international_raceway)[0],
                            fairbanks_alaska = poly_databucket.intersects(poly_fairbanks_alaska)[0],
                        )
                        intersections = {
                            k: bool(v)
                            for k, v in intersections.items()
                        }
                        location_list = [key for key, value in intersections.items() if value is True]
                        if len(location_list) > 0:
                            location = ','.join(location_list)
                        else:
                            location = None
                        return location 

                    if latitude != "n/a" or longitude != "n/a":
                        try:
                            location = find_geo_location_name(latitude, longitude)
                            print(f"Location of trigger: {location}")
                            table_lat = float(np.float64(latitude))
                            table_lon = float(np.float64(longitude))
                        except Exception as e:
                            tqdm.write("Failed to find location name from coordinates")
                            tqdm.write(f"Error: {e}")
                            sys.stdout.flush()
                            location = "n/a"
                    else:
                        tqdm.write("No coordinates found for this trigger.")
                        sys.stdout.flush()
                        location = "n/a"
                    #----------------------------------------------------------------------------------------------

                    #----------------- Insert findings into SQL tables -------------------
                    trigger_status = "successful population"
                    table_package_filepath = package_filepath.replace("/data030", r"\\stnafcvdo030.us164.corpintra.net\ntm").replace("/", "\\")
                    # Insert into data.trigger
                    try:
                        cur.execute("""
                            INSERT INTO data.trigger (
                                databucketid, trigger_filepath, wav_filename, mf4_filename,
                                hostcam1_jpg, hostcam2_jpg, hostcam3_jpg, hostcam4_jpg,
                                hostcam5_jpg, hostcam6_jpg, hostcam7_jpg, hostcam8_jpg,
                                hostcam1_mp4, hostcam2_mp4, hostcam3_mp4, hostcam4_mp4,
                                hostcam5_mp4, hostcam6_mp4, hostcam7_mp4, hostcam8_mp4,
                                transcription_text, trigger_status, trigger_package_creation_time, truck_name, trigger_latitude, trigger_longitude, geolocation
                            )
                            VALUES (
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s
                            )
                            RETURNING triggerid;
                        """, (
                            databucket_id, table_package_filepath,
                            pulled_files_from_trigger_folder[trigger_package][".wav"],
                            pulled_files_from_trigger_folder[trigger_package][".mf4"],
                            jpgs["hostcam1_jpg"], jpgs["hostcam2_jpg"], jpgs["hostcam3_jpg"], jpgs["hostcam4_jpg"],
                            jpgs["hostcam5_jpg"], jpgs["hostcam6_jpg"], jpgs["hostcam7_jpg"], jpgs["hostcam8_jpg"],
                            mp4s["hostcam1_mp4"], mp4s["hostcam2_mp4"], mp4s["hostcam3_mp4"], mp4s["hostcam4_mp4"],
                            mp4s["hostcam5_mp4"], mp4s["hostcam6_mp4"], mp4s["hostcam7_mp4"], mp4s["hostcam8_mp4"],
                            transcribed_text, trigger_status, formatted_timestamp, truck_name, table_lat, table_lon, location
                        ))

                        # Fetch the returned triggerid
                        trigger_id = cur.fetchone()[0]
                        tqdm.write(f"Trying to commit for databucket {databucket_id}")
                        sys.stdout.flush()
                        conn.commit()
                        cur.execute("UPDATE data.trigger_jobqueue SET jobstatus = %s WHERE databucketid =%s ",("databucket processed; contains triggers",databucket_id))
                        conn.commit()
                        tqdm.write(f"data.trigger commit successful for databucketid:{databucket_id}, triggerid:{trigger_id}.")
                    except Exception as e:
                        tqdm.write("Commit failed:", e)
                        tqdm.write(f"Error: {e}")
                        sys.stdout.flush()
                        trigger_status = "populating error"
                        conn.rollback()
                        cur.execute("UPDATE data.trigger_jobqueue SET jobstatus = %s WHERE databucketid =%s ",("previous attempted databucket process failed",databucket_id))
                        conn.commit()
                    
                    string = "untagged"
                    try:
                        if transcribed_text != "" and trigger_status == "successful population":
                            cur.execute("""
                                INSERT INTO data.tags (databucketid, triggerid, tagging_status)
                                VALUES (%s, %s, %s)
                                RETURNING tagid;
                            """, (databucket_id, trigger_id, string))
                            tag_id = cur.fetchone()[0]
                            conn.commit()
                            tqdm.write(f"data.tags commit successful for databucketid:{databucket_id}, triggerid:{trigger_id}, tagid:{tag_id}.")
                            sys.stdout.flush()
                    except Exception as e:
                        tqdm.write("Commit failed:", e)
                        tqdm.write(f"Error: {e}")
                        sys.stdout.flush()
                        conn.rollback()
                        cur.execute("INSERT INTO data.tags(tagging_status) VALUES (%s)", ("error with data.trigger",))
                        conn.commit()
            count += 1
    except KeyboardInterrupt:
         tqdm.write("Shutting down.")
         sys.stdout.flush()
    cur.close()
    conn.close()
    cur.close()
    conn.close()
    #---------------------------------------------------------

if __name__ == "__main__":
    main()
