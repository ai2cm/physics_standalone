if [ ! -d serialbox ]; then
  git clone --single-branch --branch savepoint_as_string https://github.com/VulcanClimateModeling/serialbox2.git serialbox
fi

docker build -t phys_standalone .
