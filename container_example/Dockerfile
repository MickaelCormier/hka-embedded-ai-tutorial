FROM --platform=linux/arm64 jetson_gstreamer:jetpack4

# Copy and build
WORKDIR /app
COPY gstreamer gstreamer

RUN mkdir /app/gstreamer/cpp/build && cd /app/gstreamer/cpp/build && \
    cmake .. && make -j 4

RUN cd /app/gstreamer/cpp && pip install .

CMD [ "/bin/bash" ]
