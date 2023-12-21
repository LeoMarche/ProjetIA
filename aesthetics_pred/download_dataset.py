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

DOWNLOAD_URL_DATA = "https://drive.usercontent.google.com/download?id=1FAm4qfbhoflEtNCw90OjM7WlEl7brlWb&export=download&confirm=t&at=APZUnTVMDNqFFoAE60ucthsYxLHZ:1703156399392"

# Replace me
DOWNLOAD_FOLDER = "./dataset"

if not(os.path.isdir(DOWNLOAD_FOLDER)):
    os.mkdir(DOWNLOAD_FOLDER)
    os.mkdir(os.path.join(DOWNLOAD_FOLDER, 'data'))

download_file(DOWNLOAD_URL_DATA, os.path.join(DOWNLOAD_FOLDER, "images.zip"))

with zipfile.ZipFile(os.path.join(DOWNLOAD_FOLDER, "images.zip"), 'r') as zip_ref:
    zip_ref.extractall(os.path.join(DOWNLOAD_FOLDER, 'data'))
