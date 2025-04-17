urls=(
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_0001_1000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_1001_2000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_2001_3000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_3001_4000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_4001_5000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_5001_6000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_6001_7000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_7001_8000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_8001_9000.zip"
  "https://data.vertlife.org/birdtree/Stage1/HackettStage1_9001_10000.zip"
)
mkdir downloads
for url in "${urls[@]}"; do
  curl -L -o "downloads/$(basename $url)" "$url"
done
mkdir -p extracted
mkdir -p output
for file in downloads/*.zip; do
  echo "Unzipping $file..."
  unzip -o "$file" -d extracted/
done