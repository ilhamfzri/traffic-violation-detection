wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1b30tspvv7pPfj3e826hFlDIlM5ojLJ4v' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1b30tspvv7pPfj3e826hFlDIlM5ojLJ4v" -O model_v2.tar && rm -rf /tmp/cookies.txt
tar -xvf model_v2.tar