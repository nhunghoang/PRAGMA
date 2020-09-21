#!/bin/bash

# end every server process 
SERVER_PIDS="server_pids.txt"
while IFS= read -r line
do 
    kill ${line}
done < ${SERVER_PIDS}
deactivate pragma_venv
printf "\nPRAGMA has been successfully shut down.\n\n"

