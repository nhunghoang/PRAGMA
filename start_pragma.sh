#!/bin/bash

# pid text file needed for ending PRAGMA
echo -n > server_pids.txt

# check that python3 is installed
if ! type -P python3 >/dev/null 2>&1 
then 
    printf "\nWARNING: python3 not found. Install and try again.\n\n"
else
    # check that pragma_venv is activated
    VENV=$(python3 -c 'import os; print("VIRTUAL_ENV" in os.environ)')
    if [ "${VENV}" = "False" ]
    then 
        printf "\nWARNING: pragma_venv not activated. Activate and try again.\n\n"
    else
        # run PRAGMA venv and servers
        python3 web_server.py &> /dev/null &  
        python3 simple_server.py &> server_messages.log &
        printf "\nPRAGMA is ready to run on Observable.\n\n"
    fi
fi

