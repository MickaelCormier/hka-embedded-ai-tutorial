FROM --platform=linux/arm64 container_gstreamer

# Copy and build
WORKDIR /app
COPY gstreamer gstreamer

RUN mkdir /app/gstreamer/build && cd /app/gstreamer/build && \
    cmake .. && make -j 4

RUN cd /app/gstreamer && pip install .

CMD [ "/bin/bash" ]
