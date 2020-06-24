if [ ! -d serialbox ]; then
  git clone -b master https://github.com/GridTools/serialbox.git serialbox
  cd serialbox
  git checkout -b savepoint_as_string
  cd -
fi

docker build -t phys_standalone .
