import os
import subprocess

def run(cmd):
    subprocess.call(cmd, shell=True)

def menu():
    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("============================================")
        print("        Scary Story Bot - Master Menu       ")
        print("============================================")
        print("1. Start Bot")
        print("2. Download Background Videos")
        print("3. Download Sounds (jumpscares + ambience)")
        print("4. Backup Project")
        print("Q. Quit")
        choice = input("\nEnter choice: ").strip().lower()

        if choice == "1":
            run("STARTBOT.bat")
        elif choice == "2":
            run("DOWNLOAD_VIDEOS.bat")
        elif choice == "3":
            run("DOWNLOAD_SOUNDS.bat")
        elif choice == "4":
            run("BACKUP.bat")
        elif choice == "q":
            break

if __name__ == "__main__":
    menu()
Paste your patch below. Press CTRL+Z then Enter when finished:
Paste your patch below. Press CTRL+Z then Enter when finished:
