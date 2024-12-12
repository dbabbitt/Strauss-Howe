#!/bin/bash

# Navigate to the submodule directory
cd share || exit

# Check for changes in the submodule
if [[ $(git status --porcelain) ]]; then
    echo "Changes detected in the submodule. Resetting to the latest commit on origin/master..."
    git reset --hard origin/master
fi

# Return to the main repository
cd ..

# Allow the commit to proceed
exit 0