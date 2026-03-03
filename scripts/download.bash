# Create target dir
mkdir -p ../data

# 2017 Train images [118K/18GB]
curl -L --fail --retry 10 --retry-delay 3 \
  -o ../data/train2017.zip \
  http://images.cocodataset.org/zips/train2017.zip

# 2017 Val images [5K/1GB]
curl -L --fail --retry 10 --retry-delay 3 \
  -o ../data/val2017.zip \
  http://images.cocodataset.org/zips/val2017.zip

# 2017 Test images [41K/6GB]
curl -L --fail --retry 10 --retry-delay 3 \
  -o ../data/test2017.zip \
  http://images.cocodataset.org/zips/test2017.zip