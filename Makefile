IMAGE_NAME ?= phys_standalone
IMAGE_TAG ?= latest
GCR_URL = us.gcr.io/vcm-ml
IMAGE = $(GCR_URL)/$(IMAGE_NAME):$(IMAGE_TAG)

build:
	if [ ! -d serialbox ]; then \
		git clone --single-branch --branch savepoint_as_string https://github.com/VulcanClimateModeling/serialbox2.git serialbox; \
	fi

	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) -t $(IMAGE) .

push:
	docker push $(IMAGE)

test:
	docker run -v serial_convert:/serial_convert -w=/ -it $(IMAGE) /tests/test_to_zarr.sh

.PHONY: build push test