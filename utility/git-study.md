Git my study



## Git flow in action

1. `git flow init`

2. `git push origin --all`: push to remote repository 

3. `git flow feature start <feature name>`: start a new feature

   #### During develop process, often use:

4. `git add .` : prepare  which files for commit

5. `git commit -m <commit message>`

   #### When feature completed

6. `git flow feature finish`: finish a feature (on that brand)

   #### Release (product version)

7. `git flow release start <version>`

8. `git flow release finish`

9. `git checkout master`

10. `git push origin --all --follow-tags `

    #### Some more comment

11. `git branch` : list out all branch

## git MERGE vs REBASE

1. checkout to the branch that you want to rebase to master (source branch): `git checkout <branch's name>`

   <img src="/home/adidaphat/Downloads/rebase_idea.png" alt="rebase_idea" style="zoom:80%;" />

2. on source brand: `git rebase <destination branch>`
