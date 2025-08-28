#!/usr/bin/env python3

import os
import subprocess

# List of all commit hashes
commits = [
    "76dd607", "a903853", "804add6", "56210f5", "38df887", "aa250d0",
    "1b74ca8", "ce0a526", "73984a8", "9e40b3f", "9e106d8", "56c34dd",
    "52e2288", "0cfbe90", "449f665", "4383646", "e43e4b9", "f527eb1",
    "afa3708", "4144921", "113c4db", "534c0be", "8955aa7", "2aeb68f",
    "c186c80", "0b023c8", "90f4e31", "5b1a6f8", "77fe28b", "405127d",
    "5e13357", "d86f5b9", "dc8f883", "780cc26", "b7516bb", "85506b4",
    "1172e63", "aecc646", "ed4b16c", "8bb5be5", "f6aa225", "16a674c",
    "14a0d9c", "b6abe0f", "e550d50", "3b469e8", "45eda16", "d5be59f",
    "fdaed11", "dd69e2d", "30e8897", "7070a07", "4b4288f", "03e27b4",
    "8eb9e60", "f1b1c2f", "763b964", "40bf5de", "28fbe4f", "21c05a2",
    "825c8e8", "65a2517", "a33a314", "7834194", "8b44897", "1604b02",
    "7b4e7bc", "4d0a72c", "fa1cb9b", "875af7c", "b4c3432", "57a8a1f",
    "ddf4d8a", "9582393", "f382103"
]

def print_individual_git_show_commands(commits):
    print("=== Individual git show commands ===")
    for commit in commits:
        print(f"git show {commit}")

def save_all_diffs_to_files(commits, outdir="commit_diffs"):
    print("\n=== Saving all diffs to files ===")
    os.makedirs(outdir, exist_ok=True)
    for commit in commits:
        print(f"Fetching diff for commit: {commit}")
        outpath = os.path.join(outdir, f"{commit}.diff")
        try:
            with open(outpath, "w") as f:
                subprocess.run(["git", "show", commit], stdout=f, stderr=subprocess.PIPE, check=True)
            print(f"✓ Saved diff to {outpath}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to fetch diff for {commit}: {e}")

    print(f"All diffs saved to {outdir}/ directory")

def save_all_diffs_to_single_file(commits, outfile="all_commit_diffs.txt"):
    print("\n=== Saving all diffs to a single file ===")
    with open(outfile, "w") as f:
        for commit in commits:
            f.write(f"=== COMMIT {commit} ===\n")
            try:
                result = subprocess.run(["git", "show", commit], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
                f.write(result.stdout)
            except subprocess.CalledProcessError as e:
                f.write(f"Failed to fetch diff for {commit}: {e}\n")
            f.write("\n")
    print(f"All diffs saved to {outfile}")

def print_git_log_patch_range(start_commit, end_commit, outfile="all_diffs_in_range.txt"):
    print("\n=== Alternative: Use git log with patch format ===")
    print("# Get all commits in range with full diffs:")
    print(f"git log --patch --reverse {start_commit}..{end_commit} > {outfile}")

def print_git_show_specific_commits(commits, outfile="all_specific_commits.diff"):
    print("\n# Or get specific commits only:")
    commit_str = " ".join(commits)
    print(f"git show {commit_str} > {outfile}")

if __name__ == "__main__":
    print_individual_git_show_commands(commits)
    save_all_diffs_to_files(commits)
    save_all_diffs_to_single_file(commits)
    # Print alternative git log and git show commands for reference
    print_git_log_patch_range("f382103", "76dd607")
    print_git_show_specific_commits(commits)
