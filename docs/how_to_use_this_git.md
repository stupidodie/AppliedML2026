# Working with the code repository

### How to get files:
If you simply want to access files in a GitHub repository, you start by "copying" them to your machine with the command: 
```bash
git clone https://github.com/AppliedMachineLearningNBI/AppliedML2026.git
```
(see installation instructions, e.g. point 3 in [install_instructions_vscode.md](install_instructions_vscode.md)).

Once this is set up, you may enter the `git pull` command anywhere in this directory to update the files to the latest versions. This is useful as we add new exercises every week. 
Note that if you changed something, it will warn you, before overwriting. 
If you want to overwrite, just remove your (changed) file. But as you modify the files with your answer, you want to rename them first to avoid accidental overwritting.


### How to work with your own GitHub repository:
If you want to make your own GitHub repository, and "push" the changes to update your repository
```bash
git add myfile.py
git commit -m "This is a commit message describing what the changes are"
git push origin main
```
`git add` are the files where you made changes and want to update on GitHub. `git commit` make the changes on your local comupter. `git push` apply the changes to GitHub.

