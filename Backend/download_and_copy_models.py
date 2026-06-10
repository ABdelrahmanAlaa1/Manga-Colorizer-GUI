import os
import sys
import shutil
import urllib.request
import time

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    tempdl_dir = os.path.join(root_dir, 'tempdl')
    networks_dir = os.path.join(root_dir, 'Backend', 'networks')

    print(f"[*] Workspace Root: {root_dir}")
    print(f"[*] Source Folder: {tempdl_dir}")
    print(f"[*] Destination Folder: {networks_dir}")

    # Create directories if they don't exist
    os.makedirs(networks_dir, exist_ok=True)

    # 1. Copy valid models from tempdl/ to Backend/networks/
    if os.path.exists(tempdl_dir):
        print("\n[*] Copying valid downloaded models from tempdl/ to Backend/networks/...")
        for filename in os.listdir(tempdl_dir):
            src_path = os.path.join(tempdl_dir, filename)
            dst_path = os.path.join(networks_dir, filename)

            if not os.path.isfile(src_path):
                continue

            # Check if it's a file type we want
            if not (filename.endswith('.safetensors') or filename.endswith('.pth') or filename.endswith('.pt')):
                continue

            # Check size to skip LFS stubs (e.g. less than 5KB)
            size_bytes = os.path.getsize(src_path)
            if size_bytes < 5000:
                print(f"[!] Skipping LFS text stub: {filename} ({size_bytes} bytes)")
                continue

            print(f"[+] Copying {filename} ({size_bytes / (1024*1024):.2f} MB)...")
            shutil.copy2(src_path, dst_path)
    else:
        print("\n[!] Source folder 'tempdl' does not exist. Skipping copy step.")

    # 2. Download missing/stub models
    downloads = [
        {
            "name": "4x-UltraSharp.pth",
            "url": "https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
        },
        {
            "name": "4x-AnimeSharp.pth",
            "url": "https://huggingface.co/utnah/esrgan/resolve/main/4x-AnimeSharp.pth"
        },
        {
            "name": "4x_eula_digimanga_bw_v2_nc1_307k.pth",
            "url": "https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_eula_digimanga_bw_v2_nc1_307k.pth"
        }
    ]

    print("\n[*] Downloading complete binary models from Hugging Face...")
    for item in downloads:
        name = item["name"]
        url = item["url"]
        dst_path = os.path.join(networks_dir, name)

        # If it already exists and is large, we can skip it
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 1000000:
            print(f"[~] {name} already exists and is valid ({os.path.getsize(dst_path) / (1024*1024):.2f} MB). Skipping download.")
            continue

        print(f"[*] Downloading {name} from {url}...")
        try:
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            )
            
            start_time = time.time()
            with urllib.request.urlopen(req) as response:
                total_size = int(response.info().get('Content-Length', 0))
                bytes_downloaded = 0
                chunk_size = 1024 * 256 # 256 KB chunks

                with open(dst_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        
                        if total_size > 0:
                            percent = (bytes_downloaded / total_size) * 100
                            mb_down = bytes_downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            sys.stdout.write(f"\r    -> Progress: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                            sys.stdout.flush()
                        else:
                            mb_down = bytes_downloaded / (1024 * 1024)
                            sys.stdout.write(f"\r    -> Downloaded: {mb_down:.1f} MB")
                            sys.stdout.flush()
                            
            duration = time.time() - start_time
            print(f"\n[+] Successfully downloaded {name} in {duration:.1f}s.")
        except Exception as e:
            print(f"\n[!] Failed to download {name}: {e}")

    print("\n[+] All model copying and downloading completed successfully!")

if __name__ == "__main__":
    main()
