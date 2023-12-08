import os, requests, zipfile

def download_file(url, local_filename):
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=32768): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                if chunk:
                    f.write(chunk)
    return local_filename

DOWNLOAD_URL_DATA = "https://drive.google.com/u/0/uc?id=1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5&export=download&confirm=1"
DOWNLOAD_URL_GROUND_TRUTH = "https://drive.google.com/a/umn.edu/uc?id=1PnO7szbdub1559LfjYHMy65EDC4VhJC8&export=download&confirm=1"

# Replace me
DOWNLOAD_FOLDER = "./dataset"

if not(os.path.isdir(DOWNLOAD_FOLDER)):
    os.mkdir(DOWNLOAD_FOLDER)
    os.mkdir(os.path.join(DOWNLOAD_FOLDER, 'data'))
    os.mkdir(os.path.join(DOWNLOAD_FOLDER, 'truth'))

download_file(DOWNLOAD_URL_DATA, os.path.join(DOWNLOAD_FOLDER, "images.zip"))
download_file(DOWNLOAD_URL_GROUND_TRUTH, os.path.join(DOWNLOAD_FOLDER, 'truth.zip'))

with zipfile.ZipFile(os.path.join(DOWNLOAD_FOLDER, "images.zip"), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(DOWNLOAD_FOLDER, 'data'))

with zipfile.ZipFile(os.path.join(DOWNLOAD_FOLDER, "truth.zip"), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(DOWNLOAD_FOLDER, 'truth'))


