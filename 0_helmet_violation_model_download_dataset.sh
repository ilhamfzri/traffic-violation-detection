wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18e7ZPor0ZtQ6Qd9ejSFMYjU6OyGtHgLz' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18e7ZPor0ZtQ6Qd9ejSFMYjU6OyGtHgLz" -O helmet_violation_dataset.tar && rm -rf /tmp/cookies.txt
tar -xvf helmet_violation_dataset.tar
mv final_helmet_violation_dataset dataset/helmet_violation_dataset/final_helmet_violation_dataset