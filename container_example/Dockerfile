FROM --platform=linux/arm64 container_gstreamer

# Copy and build
WORKDIR /app
COPY ocv_ipstream_gstreamer ocv_ipstream_gstreamer

RUN mkdir /app/ocv_ipstream_gstreamer/build && cd /app/ocv_ipstream_gstreamer/build && \
    cmake .. && make -j 4

RUN cd /app/ocv_ipstream_gstreamer && pip install .

CMD [ "/bin/bash" ]
