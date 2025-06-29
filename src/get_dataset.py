import os
import subprocess
import gzip
import shutil


def download_mimic_iv_ed_demo(output_dir="mimic-iv-ed-demo"):
    """
    Download MIMIC-IV ED demo dataset using wget.
    """
    os.makedirs(output_dir, exist_ok=True)
    url = "https://physionet.org/files/mimic-iv-ed-demo/2.2/"
    print(f"Downloading from {url} ...")
    
    command = [
        "wget",
        "-r",        
        "-N",    
        "-c",     
        "-np",    
        "-P", output_dir, 
        url
    ]
    
    subprocess.run(command, check=True)
    print("Download complete.")


def extract_gz_files(source_folder, destination_folder):
    """
    Extract all .gz files from source_folder to destination_folder.
    """
    os.makedirs(destination_folder, exist_ok=True)

    for root, _, files in os.walk(source_folder):
        for filename in files:
            if filename.endswith(".gz"):
                source_path = os.path.join(root, filename)
                dest_filename = filename[:-3] 
                dest_path = os.path.join(destination_folder, dest_filename)

                with gzip.open(source_path, 'rb') as f_in:
                    with open(dest_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                print(f"Extracted: {filename} → {dest_path}")
                
                


def delete_downloaded_folder(folder_path):
    """
    Delete the specified folder and its contents.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder does not exist: {folder_path}")

if __name__ == "__main__":
    # Download the MIMIC-IV ED demo dataset
    download_mimic_iv_ed_demo()

    # Extract the downloaded .gz files
    extract_gz_files(source_folder="mimic-iv-ed-demo/physionet.org/files/mimic-iv-ed-demo/2.2/ed", 
                     destination_folder="data/raw")
    
    # Delete the downloaded folder
    delete_downloaded_folder("mimic-iv-ed-demo")
    print("All operations completed successfully.")
