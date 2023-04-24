#!/bin/bash

SA_DYNAMICS="$(cat requirements.txt | egrep safe-autonomy-dynamics== | egrep -o '[0-9.]+' | tr -d '\n')"
RTA="$(cat requirements.txt | egrep run-time-assurance== | egrep -o '[0-9.]+' | tr -d '\n')"
CORL="$(cat requirements.txt | egrep corl== | egrep -o '[0-9.]+' | tr -d '\n')"

ACT3_PAT=
GIT_URL=
DEVELOP=false

while getopts p:u:dh flag; do
    case "${flag}" in
        p) ACT3_PAT="${OPTARG}";;
        d) DEVELOP=true;;
        u) GIT_URL=$OPTARG;;
        h) Help
           exit;;
        *) echo "usage: $0 [-p] [-u] [-h]" >&2
           exit;;
    esac
done

Help()
{
   # Display Help
   echo "Installs safe-autonomy-sims and dependencies into the active environment."
   echo
   echo "Syntax: install.sh [-p pat|-h]"
   echo "options:"
   echo "    p     Specify personal access token."
   echo "    u     Specify url of git remote host to clone repos from."
   echo "flags:"
   echo "    h     Print this help."
}

InstallReqs() {
  echo "pip installing safe-autonomy-sims requirements into environment from secure package registry"

  # clone CoRL repo outside of safe-autonomy-sims repo
  cd ..
  git clone https://oauth2:$1@$GIT_URL/act3-rl/corl.git
  cd corl
  git checkout v$CORL
  if [ $2 == true ]; then
    pip install -e .
    cd ..
  else
    pip install .
    cd ..
    rm -rf corl
  fi
  cd safe-autonomy-sims

  pip install safe-autonomy-dynamics==$SA_DYNAMICS \
  --index-url https://__token__:"$1"@$GIT_URL/api/v4/projects/826/packages/pypi/simple
  pip install run-time-assurance==$RTA \
  --index-url https://__token__:"$1"@$GIT_URL/api/v4/projects/804/packages/pypi/simple
}

InstallSims() {
  echo "pip installing safe-autonomy-sims into environment from secure package registry"

  pip install safe-autonomy-sims \
  --index-url https://__token__:"$1"@$GIT_URL/api/v4/projects/660/packages/pypi/simple
}

InstallSimsDev() {
  # Check working directory is repo root
  parentdir="$(basename "$(pwd)")"
  echo $parentdir
  echo $(pwd)
  if ! [ "$parentdir" = "safe-autonomy-sims" ]; then
    echo "Development installation must be executed inside repo root"
    exit
  fi

  echo "pip installing safe-autonomy-sims into environment from source (editable)"

  pip install -e .[dev]
}

InstallReqs "$ACT3_PAT" $DEVELOP

if [ $DEVELOP = true ]; then
  InstallSimsDev
else
  InstallSims "$ACT3_PAT"
fi
