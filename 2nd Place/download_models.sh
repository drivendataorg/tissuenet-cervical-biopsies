wget -O models.zip --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-FzuljprLaC6dm6WeegINKp9j0Vg3-Wd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-FzuljprLaC6dm6WeegINKp9j0Vg3-Wd" && rm -rf /tmp/cookies.txt
unzip -qq models.zip
mkdir -p workspace
mv models_selected workspace/models