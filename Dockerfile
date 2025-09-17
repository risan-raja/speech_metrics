FROM debian:bookworm-20250908

WORKDIR /root/work

RUN apt-get update && apt-get install -y \
  build-essential \
  cmake \
  git \
  curl

CMD ["/bin/tail", "-f", "/dev/null"]
