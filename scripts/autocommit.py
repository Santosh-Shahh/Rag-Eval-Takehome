import subprocess
import time
import os
import sys

def has_changes():
    """Checks if there are any uncommitted changes in the repository."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        # Filter out changes to this script's log or temporary watcher files to prevent infinite loops
        status_lines = result.stdout.strip().splitlines()
        filtered_lines = [l for l in status_lines if "autocommit" not in l]
        return len(filtered_lines) > 0
    except subprocess.CalledProcessError as e:
        print(f"Git status failed: {e}", file=sys.stderr)
        return False

def commit_and_push():
    """Stages, commits, and pushes changes to the remote branch."""
    print("Changes detected. Staging files...")
    subprocess.run(["git", "add", "."])
    
    print("Committing changes...")
    subprocess.run(["git", "commit", "-m", "Auto-update: local workspace changes detected"])
    
    print("Pushing to GitHub remote...")
    push_result = subprocess.run(["git", "push"], capture_output=True, text=True)
    
    if push_result.returncode == 0:
        print("Successfully auto-pushed updates to GitHub!")
    else:
        print(f"Failed to auto-push: {push_result.stderr}", file=sys.stderr)

def main():
    print("==================================================")
    print("Auto-Commit file watcher started.")
    print("Monitoring workspace for changes every 5 seconds...")
    print("Press Ctrl+C to stop.")
    print("==================================================")
    
    # Change directory to the workspace root
    workspace_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(workspace_root)
    
    while True:
        try:
            if has_changes():
                commit_and_push()
        except KeyboardInterrupt:
            print("\nAuto-Commit watcher stopped.")
            break
        except Exception as e:
            print(f"Error in watcher loop: {e}", file=sys.stderr)
            
        time.sleep(5)

if __name__ == "__main__":
    main()
