if [ ! -d serialbox ]; then
  git clone -b v2.6.1 --depth 1 https://github.com/GridTools/serialbox.git
fi

docker build -t physics_standalone .
